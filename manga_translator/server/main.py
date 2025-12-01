import io
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
from argparse import Namespace
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pathlib import Path

from manga_translator import Config
from manga_translator.server.instance import ExecutorInstance, executor_instances
from manga_translator.server.myqueue import task_queue
from manga_translator.server.request_extraction import get_ctx, while_streaming, TranslateRequest, BatchTranslateRequest, get_batch_ctx
from manga_translator.server.to_json import to_translation, TranslationResponse
from contextlib import contextmanager
import logging

logger = logging.getLogger('manga_translator.server')

# 设置Web服务器标志，防止翻译器重新加载.env覆盖用户环境变量
os.environ['MANGA_TRANSLATOR_WEB_SERVER'] = 'true'

# 启动时加载 .env 文件
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"[INFO] Loaded environment variables from: {env_path}")
    # 打印已加载的 API Keys（不显示值）
    loaded_keys = [k for k in os.environ.keys() if 'API' in k or 'KEY' in k or 'TOKEN' in k]
    if loaded_keys:
        print(f"[INFO] Loaded API keys: {', '.join(loaded_keys)}")
else:
    print(f"[WARNING] .env file not found at: {env_path}")

app = FastAPI()
nonce = None

# 全局服务器配置（从启动参数设置）
server_config = {
    'use_gpu': False,
    'use_gpu_limited': False,
    'verbose': False,
    'models_ttl': 0,
    'retry_attempts': None,
    'admin_password': None,  # 管理员密码
    'max_concurrent_tasks': 3,  # 最大并发任务数
}

# 并发控制信号量（根据max_concurrent_tasks动态创建）
translation_semaphore = None

def init_semaphore():
    """初始化并发控制信号量"""
    global translation_semaphore
    max_concurrent = server_config.get('max_concurrent_tasks', 3)
    translation_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"并发控制已初始化: 最大并发任务数 = {max_concurrent}")

# 有效的管理员 tokens（登录后生成）
valid_admin_tokens = set()

# 管理员配置文件路径
ADMIN_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'admin_config.json')

# 默认管理端配置
DEFAULT_ADMIN_SETTINGS = {
    'visible_sections': ['translator', 'cli', 'detector', 'ocr', 'inpainter', 'render', 'upscale', 'colorizer'],  # 用户端可见的配置区域（与桌面版UI一致）
    'hidden_keys': [  # 用户端隐藏的配置项（精确到参数级别）
        # 通过其他UI控制的参数
        'upscale.realcugan_model',  # 通过 upscale_ratio 动态控制
        'cli.load_text',  # 通过工作流模式下拉框控制
        'cli.template',  # 通过工作流模式下拉框控制
        'cli.generate_and_export',  # 通过工作流模式下拉框控制
        'cli.colorize_only',  # 通过工作流模式下拉框控制
        'cli.upscale_only',  # 通过工作流模式下拉框控制
        'cli.inpaint_only',  # 通过工作流模式下拉框控制
        'cli.batch_concurrent',  # 并发处理已隐藏（桌面版也隐藏）
        # 翻译后检查参数（默认隐藏）
        'translator.enable_post_translation_check',
        'translator.post_check_max_retry_attempts',
        'translator.post_check_repetition_threshold',
        'translator.post_check_target_lang_threshold',
        # 高级翻译器参数（默认隐藏）
        'translator.translator_chain',  # 链式翻译
        'translator.selective_translation',  # Selective Translation
        'translator.skip_lang',  # Skip Lang
        # 废弃参数
        'render.gimp_font',  # 已废弃，使用 font_path 代替
    ],
    'readonly_keys': [],  # 用户端只读的配置项（可见但不可修改）
    'default_values': {},  # 管理员设置的默认值（会覆盖 config.json）
    'allowed_translators': [],  # 允许用户使用的翻译器列表（空=全部允许）
    'allowed_languages': [],  # 允许用户使用的语言列表（空=全部允许）
    'allowed_workflows': [],  # 允许用户使用的翻译流程（空=全部允许）
    'permissions': {
        'can_upload_fonts': True,  # 允许上传字体
        'can_delete_fonts': True,  # 允许删除字体
        'can_upload_prompts': True,  # 允许上传提示词
        'can_delete_prompts': True,  # 允许删除提示词
        'can_add_folders': True,  # 允许添加文件夹（批量翻译）
    },
    'upload_limits': {
        'max_image_size_mb': 10,  # 单张图片最大大小（MB），0表示不限制
        'max_images_per_batch': 50,  # 一次最多上传图片数量，0表示不限制
    },
    'user_access': {
        'require_password': False,  # 是否需要密码访问用户端
        'user_password': '',  # 用户端密码（空表示不需要密码）
    },
    'api_key_policy': {
        'require_user_keys': False,  # 是否强制用户提供自己的 API Key
        'allow_server_keys': True,  # 是否允许使用服务器的 API Key
        'save_user_keys_to_server': False,  # 是否将用户的 API Key 保存到服务器 .env
    },
    'show_env_to_users': False,  # 是否在用户端显示 .env 编辑功能
    'announcement': {
        'enabled': False,  # 是否启用公告
        'message': '',  # 公告内容
        'type': 'info',  # 公告类型：info, warning, error
    },
}

def load_admin_settings():
    """从文件加载管理员配置"""
    if os.path.exists(ADMIN_CONFIG_PATH):
        try:
            with open(ADMIN_CONFIG_PATH, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
                print(f"[INFO] Loaded admin settings from: {ADMIN_CONFIG_PATH}")
                # 合并默认配置和加载的配置
                settings = DEFAULT_ADMIN_SETTINGS.copy()
                settings.update(loaded_settings)
                
                # 如果配置文件中没有密码，尝试从环境变量读取
                if not settings.get('admin_password'):
                    env_password = os.environ.get('MANGA_TRANSLATOR_ADMIN_PASSWORD')
                    if env_password and len(env_password) >= 6:
                        settings['admin_password'] = env_password
                        # 保存到配置文件
                        save_admin_settings(settings)
                        print(f"[INFO] Admin password set from environment variable MANGA_TRANSLATOR_ADMIN_PASSWORD")
                    elif env_password:
                        print(f"[WARNING] MANGA_TRANSLATOR_ADMIN_PASSWORD is too short (minimum 6 characters)")
                
                return settings
        except Exception as e:
            print(f"[ERROR] Failed to load admin settings: {e}")
            return DEFAULT_ADMIN_SETTINGS.copy()
    else:
        print(f"[INFO] Admin config file not found, using defaults: {ADMIN_CONFIG_PATH}")
        settings = DEFAULT_ADMIN_SETTINGS.copy()
        
        # 首次启动时，尝试从环境变量读取密码
        env_password = os.environ.get('MANGA_TRANSLATOR_ADMIN_PASSWORD')
        if env_password and len(env_password) >= 6:
            settings['admin_password'] = env_password
            # 保存到配置文件
            save_admin_settings(settings)
            print(f"[INFO] Admin password set from environment variable MANGA_TRANSLATOR_ADMIN_PASSWORD")
        elif env_password:
            print(f"[WARNING] MANGA_TRANSLATOR_ADMIN_PASSWORD is too short (minimum 6 characters)")
        
        return settings

def save_admin_settings(settings):
    """保存管理员配置到文件"""
    try:
        with open(ADMIN_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved admin settings to: {ADMIN_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save admin settings: {e}")
        return False

# 加载管理端配置
admin_settings = load_admin_settings()

# 所有可用的翻译流程
AVAILABLE_WORKFLOWS = [
    'normal',  # 正常翻译流程
    'export_trans',  # 导出翻译
    'export_raw',  # 导出原文
    'import_trans',  # 导入翻译并渲染
    'colorize',  # 仅上色
    'upscale',  # 仅超分
    'inpaint',  # 仅修复
]

# 服务器配置文件路径（不使用 examples/config.json，避免冲突）
SERVER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'server_config.json')
# 如果服务器配置不存在，从模板配置复制一份作为初始配置
if not os.path.exists(SERVER_CONFIG_PATH):
    EXAMPLE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'config-example.json')
    if os.path.exists(EXAMPLE_CONFIG_PATH):
        import shutil
        shutil.copy(EXAMPLE_CONFIG_PATH, SERVER_CONFIG_PATH)
        print(f"[INFO] Created server config from template: {SERVER_CONFIG_PATH}")
    else:
        print(f"[WARNING] Template config not found, will use default Config()")

DEFAULT_CONFIG_PATH = SERVER_CONFIG_PATH

def load_default_config_dict():
    """加载默认配置文件，返回字典格式（包含Qt UI的完整配置）"""
    if os.path.exists(DEFAULT_CONFIG_PATH):
        try:
            with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return config_dict
        except Exception as e:
            print(f"[WARNING] Failed to load default config from {DEFAULT_CONFIG_PATH}: {e}")
            return {}
    else:
        print(f"[WARNING] Default config file not found: {DEFAULT_CONFIG_PATH}")
        return {}

def load_default_config() -> Config:
    """加载默认配置文件，返回Config对象"""
    config_dict = load_default_config_dict()
    if config_dict:
        try:
            return Config.parse_obj(config_dict)
        except Exception as e:
            print(f"[WARNING] Failed to parse config: {e}")
            return Config()
    return Config()

def parse_config(config_str: str) -> Config:
    """解析配置，如果为空则使用默认配置"""
    if not config_str or config_str.strip() in ('{}', ''):
        print("[INFO] No config provided, using default config from examples/config.json")
        return load_default_config()
    else:
        return Config.parse_raw(config_str)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加验证错误处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误，返回详细的错误信息"""
    error_details = []
    for error in exc.errors():
        error_details.append({
            'loc': error['loc'],
            'msg': error['msg'],
            'type': error['type']
        })
    
    print(f"[ERROR] Request validation failed for {request.url.path}")
    print(f"[ERROR] Validation errors: {json.dumps(error_details, indent=2)}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": error_details,
            "body": str(exc.body) if hasattr(exc, 'body') else None
        }
    )

# 添加静态文件服务
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 添加result文件夹静态文件服务
if os.path.exists("../result"):
    app.mount("/result", StaticFiles(directory="../result"), name="result")

# 简单的 i18n 实现 - 直接读取桌面版的语言文件
# 获取桌面版 locales 目录的路径
desktop_locales_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'desktop_qt_ui', 'locales')
desktop_locales_dir = os.path.abspath(desktop_locales_dir)

# 全局翻译字典
translations_cache = {}

def load_translation(locale: str) -> dict:
    """加载指定语言的翻译文件"""
    if locale in translations_cache:
        return translations_cache[locale]
    
    locale_file = os.path.join(desktop_locales_dir, f"{locale}.json")
    if os.path.exists(locale_file):
        try:
            with open(locale_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                translations_cache[locale] = translations
                print(f"[INFO] Loaded {len(translations)} translations for {locale}")
                return translations
        except Exception as e:
            print(f"[ERROR] Failed to load translation file {locale_file}: {e}")
            return {}
    else:
        print(f"[WARNING] Translation file not found: {locale_file}")
        return {}

def get_available_locales() -> dict:
    """获取可用的语言列表"""
    locales = {}
    if os.path.exists(desktop_locales_dir):
        for filename in os.listdir(desktop_locales_dir):
            if filename.endswith('.json'):
                locale_code = filename[:-5]  # 移除 .json
                locales[locale_code] = locale_code
    return locales

print(f"[INFO] i18n locales directory: {desktop_locales_dir}")
print(f"[INFO] Available locales: {list(get_available_locales().keys())}")

# 文件上传目录
from manga_translator.utils import BASE_PATH

# 字体目录（使用系统的 fonts 目录）
FONTS_DIR = os.path.join(BASE_PATH, 'fonts')
os.makedirs(FONTS_DIR, exist_ok=True)

# 提示词目录（使用系统的 dict 目录）
PROMPTS_DIR = os.path.join(BASE_PATH, '..', 'dict')
PROMPTS_DIR = os.path.abspath(PROMPTS_DIR)
os.makedirs(PROMPTS_DIR, exist_ok=True)

print(f"[INFO] Fonts directory: {FONTS_DIR}")
print(f"[INFO] Prompts directory: {PROMPTS_DIR}")

@app.get("/", response_class=HTMLResponse, tags=["web"])
async def read_root():
    """Serve the Web UI index page (User mode)"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    return HTMLResponse("<h1>Web UI not installed</h1><p>Please ensure index.html exists in manga_translator/server/static/</p>")

@app.get("/admin", response_class=HTMLResponse, tags=["web"])
async def read_admin():
    """Serve the Admin UI index page"""
    admin_path = os.path.join(static_dir, "admin.html")
    if os.path.exists(admin_path):
        with open(admin_path, 'r', encoding='utf-8') as f:
            return f.read()
    return HTMLResponse("<h1>Admin UI not installed</h1><p>Please ensure admin.html exists in manga_translator/server/static/</p>")

@app.get("/config", tags=["web"])
async def get_config(mode: str = 'user'):
    """Get default configuration structure"""
    config_dict = load_default_config_dict()
    
    # 如果是用户模式，根据管理端设置过滤
    if mode == 'user':
        filtered_config = {}
        visible_sections = admin_settings.get('visible_sections', [])
        hidden_keys = admin_settings.get('hidden_keys', [])
        default_values = admin_settings.get('default_values', {})
        
        for section, content in config_dict.items():
            if isinstance(content, dict):
                # 这是一个配置区域（如 translator, detector, cli 等）
                # 跳过不在可见列表中的区域
                if visible_sections and section not in visible_sections:
                    continue
                
                filtered_content = {}
                for key, value in content.items():
                    full_key = f"{section}.{key}"
                    if full_key not in hidden_keys:
                        # 使用管理员设置的默认值（如果有）
                        filtered_content[key] = default_values.get(full_key, value)
                if filtered_content:
                    filtered_config[section] = filtered_content
            else:
                # 这是顶层参数（如 filter_text, kernel_size, mask_dilation_offset 等）
                # 顶层参数不受 visible_sections 限制，只检查是否在隐藏列表中
                if section not in hidden_keys:
                    filtered_config[section] = content
        
        return filtered_config
    
    return config_dict

@app.get("/config/structure", tags=["admin"])
async def get_config_structure(token: str = Header(alias="X-Admin-Token", default=None)):
    """Get full configuration structure with metadata (admin only)"""
    from manga_translator.config import Renderer, Alignment, Direction, InpaintPrecision
    from manga_translator.upscaling import Upscaler
    from manga_translator.translators import Translator
    from manga_translator.detection import Detector
    from manga_translator.colorization import Colorizer
    from manga_translator.inpainting import Inpainter
    from manga_translator.ocr import Ocr
    
    config_dict = load_default_config_dict()
    
    # 获取字体列表
    fonts = []
    if os.path.exists(FONTS_DIR):
        fonts = sorted([f for f in os.listdir(FONTS_DIR) if f.lower().endswith(('.ttf', '.otf', '.ttc'))])
    
    # 获取提示词列表
    prompts = []
    if os.path.exists(PROMPTS_DIR):
        prompts = sorted([f for f in os.listdir(PROMPTS_DIR) 
                         if f.lower().endswith('.json') and f not in ['system_prompt_hq.json', 'system_prompt_line_break.json']])
    
    # 定义参数选项（枚举类型）
    param_options = {
        'renderer': [member.value for member in Renderer],
        'alignment': [member.value for member in Alignment],
        'direction': [member.value for member in Direction],
        'upscaler': [member.value for member in Upscaler],
        'translator': [member.value for member in Translator],
        'detector': [member.value for member in Detector],
        'colorizer': [member.value for member in Colorizer],
        'inpainter': [member.value for member in Inpainter],
        'inpainting_precision': [member.value for member in InpaintPrecision],
        'ocr': [member.value for member in Ocr],
        'secondary_ocr': [member.value for member in Ocr],
        'upscale_ratio': ['不使用', '2', '3', '4'],
        'realcugan_model': [
            '2x-conservative', '2x-conservative-pro', '2x-no-denoise',
            '2x-denoise1x', '2x-denoise2x', '2x-denoise3x', '2x-denoise3x-pro',
            '3x-conservative', '3x-conservative-pro', '3x-no-denoise', '3x-no-denoise-pro',
            '3x-denoise3x', '3x-denoise3x-pro',
            '4x-conservative', '4x-no-denoise', '4x-denoise3x'
        ],
        'font_path': fonts,
        'high_quality_prompt_path': prompts,
        'layout_mode': ['default', 'smart_scaling', 'strict', 'fixed_font', 'disable_all', 'balloon_fill']
    }
    
    # 构建配置结构，包含每个参数的元数据
    structure = {}
    for section, content in config_dict.items():
        if isinstance(content, dict):
            structure[section] = {}
            for key, value in content.items():
                full_key = f"{section}.{key}"
                structure[section][key] = {
                    'value': value,
                    'type': type(value).__name__,
                    'full_key': full_key,
                    'hidden': full_key in admin_settings.get('hidden_keys', []),
                    'readonly': full_key in admin_settings.get('readonly_keys', []),
                    'default_override': admin_settings.get('default_values', {}).get(full_key),
                    'options': param_options.get(key)  # 添加选项列表
                }
        else:
            structure[section] = {
                'value': content,
                'type': type(content).__name__
            }
    
    return structure

@app.get("/fonts", tags=["web"])
async def get_fonts():
    """List available fonts"""
    from manga_translator.utils import BASE_PATH
    fonts_dir = os.path.join(BASE_PATH, 'fonts')
    fonts = []
    if os.path.exists(fonts_dir):
        for f in os.listdir(fonts_dir):
            if f.lower().endswith(('.ttf', '.otf', '.ttc')):
                fonts.append(f)
    return sorted(fonts)

@app.get("/translators", tags=["web"])
async def get_translators(mode: str = 'user'):
    """Get all available translators"""
    from manga_translator.translators import TRANSLATORS
    all_translators = [str(t) for t in TRANSLATORS]
    
    # 如果是用户模式且管理员设置了允许的翻译器列表
    if mode == 'user' and admin_settings.get('allowed_translators'):
        allowed = admin_settings['allowed_translators']
        return [t for t in all_translators if t in allowed]
    
    return all_translators

@app.get("/languages", tags=["web"])
async def get_languages(mode: str = 'user'):
    """Get all valid languages"""
    from manga_translator.translators import VALID_LANGUAGES
    all_languages = list(VALID_LANGUAGES)
    
    # 如果是用户模式且管理员设置了允许的语言列表
    if mode == 'user' and admin_settings.get('allowed_languages'):
        allowed = admin_settings['allowed_languages']
        return [lang for lang in all_languages if lang in allowed]
    
    return all_languages

@app.get("/workflows", tags=["web"])
async def get_workflows(mode: str = 'user'):
    """Get all available workflows"""
    # 如果是用户模式且管理员设置了允许的流程列表
    if mode == 'user' and admin_settings.get('allowed_workflows'):
        allowed = admin_settings['allowed_workflows']
        return [wf for wf in AVAILABLE_WORKFLOWS if wf in allowed]
    
    return AVAILABLE_WORKFLOWS

@app.get("/user/settings", tags=["web"])
async def get_user_settings():
    """Get user-side visibility settings"""
    permissions = admin_settings.get('permissions', {})
    api_key_policy = admin_settings.get('api_key_policy', {})
    upload_limits = admin_settings.get('upload_limits', {})
    return {
        'show_env_editor': admin_settings.get('show_env_to_users', False),
        'can_upload_fonts': permissions.get('can_upload_fonts', True),
        'can_upload_prompts': permissions.get('can_upload_prompts', True),
        'allow_server_keys': api_key_policy.get('allow_server_keys', True),
        'max_image_size_mb': upload_limits.get('max_image_size_mb', 0),
        'max_images_per_batch': upload_limits.get('max_images_per_batch', 0)
    }

@app.get("/config/options", tags=["web"])
async def get_config_options():
    """Get options for parameters that should be dropdowns"""
    from manga_translator.config import Renderer, Alignment, Direction, InpaintPrecision
    from manga_translator.upscaling import Upscaler
    from manga_translator.translators import Translator, VALID_LANGUAGES
    from manga_translator.detection import Detector
    from manga_translator.colorization import Colorizer
    from manga_translator.inpainting import Inpainter
    from manga_translator.ocr import Ocr
    from manga_translator.utils import BASE_PATH
    
    # 获取字体列表
    fonts = []
    if os.path.exists(FONTS_DIR):
        fonts = sorted([f for f in os.listdir(FONTS_DIR) if f.lower().endswith(('.ttf', '.otf', '.ttc'))])
    
    # 获取提示词列表（使用与 /prompts 端点相同的目录）
    prompts = []
    dict_dir = os.path.join(BASE_PATH, 'dict')
    if os.path.exists(dict_dir):
        prompts = sorted([f for f in os.listdir(dict_dir) 
                         if f.lower().endswith('.json') and f not in ['system_prompt_hq.json', 'system_prompt_line_break.json']])
    
    return {
        'renderer': [member.value for member in Renderer],
        'alignment': [member.value for member in Alignment],
        'direction': [member.value for member in Direction],
        'upscaler': [member.value for member in Upscaler],
        'detector': [member.value for member in Detector],
        'colorizer': [member.value for member in Colorizer],
        'inpainter': [member.value for member in Inpainter],
        'inpainting_precision': [member.value for member in InpaintPrecision],
        'ocr': [member.value for member in Ocr],
        'secondary_ocr': [member.value for member in Ocr],
        'translator': [member.value for member in Translator],
        'target_lang': list(VALID_LANGUAGES),
        'upscale_ratio': ['不使用', '2', '3', '4'],
        'realcugan_model': [
            '2x-conservative', '2x-conservative-pro', '2x-no-denoise',
            '2x-denoise1x', '2x-denoise2x', '2x-denoise3x', '2x-denoise3x-pro',
            '3x-conservative', '3x-conservative-pro', '3x-no-denoise', '3x-no-denoise-pro',
            '3x-denoise3x', '3x-denoise3x-pro',
            '4x-conservative', '4x-no-denoise', '4x-denoise3x'
        ],
        'font_path': fonts,
        'high_quality_prompt_path': prompts,
        'layout_mode': ['default', 'smart_scaling', 'strict', 'fixed_font', 'disable_all', 'balloon_fill']
    }

# --- API Keys Management ---

@app.get("/translator-config/{translator}", tags=["web"])
async def get_translator_config(translator: str):
    """Get translator configuration (required API keys) - public info only"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'config', 'translators.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
                config = configs.get(translator, {})
                # 只返回公开信息，不返回验证规则等敏感信息
                return {
                    'name': config.get('name'),
                    'display_name': config.get('display_name'),
                    'required_env_vars': config.get('required_env_vars', []),
                    'optional_env_vars': config.get('optional_env_vars', [])
                }
        except Exception as e:
            return {}
    return {}

@app.get("/env-vars", tags=["admin"])
async def get_env_vars(token: str = Header(alias="X-Admin-Token"), show_values: bool = False):
    """Get current environment variables (admin only)"""
    from dotenv import dotenv_values
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    
    if os.path.exists(env_path):
        env_vars = dotenv_values(env_path)
        if show_values:
            # 管理员可以看到实际值
            return {
                'path': env_path,
                'vars': {key: value for key, value in env_vars.items()}
            }
        else:
            # 只返回是否已设置
            return {key: bool(value.strip()) for key, value in env_vars.items()}
    return {'path': env_path, 'vars': {}}

@app.post("/env-vars", tags=["admin"])
async def save_env_vars(env_vars: dict, token: str = Header(alias="X-Admin-Token", default=None)):
    """Save environment variables to .env file (admin only)"""
    from dotenv import set_key, load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    
    try:
        # 保存到 .env 文件
        for key, value in env_vars.items():
            if value:  # 只保存非空值
                set_key(env_path, key, value)
                # 立即更新到 os.environ，使其生效
                os.environ[key] = value
            else:
                # 如果值为空，从 .env 删除（如果存在）
                try:
                    # set_key 不支持删除，需要手动处理
                    pass
                except:
                    pass
        
        # 重新加载 .env 文件以确保所有变量都是最新的
        load_dotenv(env_path, override=True)
        
        logger.info(f"Environment variables updated: {list(env_vars.keys())}")
        return {"success": True, "message": "环境变量已更新并立即生效"}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to save env vars: {str(e)}")

@app.get("/env", tags=["web"])
async def get_user_env_vars():
    """Get environment variables for users (based on policy)"""
    from dotenv import dotenv_values
    
    # 检查是否允许用户查看 .env 编辑器
    # 注意：这只控制用户是否能"看到"API Keys，不控制翻译时是否使用服务器的 Key
    show_env_to_users = admin_settings.get('show_env_to_users', False)
    if not show_env_to_users:
        # 不显示 API Keys 编辑器给用户，返回空
        return {}
    
    # 如果显示编辑器，返回服务器的 API Keys 值（让用户可以看到和编辑）
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(env_path):
        env_vars = dotenv_values(env_path)
        return {key: value for key, value in env_vars.items() if value}
    return {}

@app.post("/env", tags=["web"])
async def save_user_env_vars(env_vars: dict):
    """Save user's environment variables"""
    from dotenv import set_key
    
    # 检查是否允许用户编辑 .env
    show_env_to_users = admin_settings.get('show_env_to_users', False)
    if not show_env_to_users:
        # 不允许用户编辑 .env，返回错误
        raise HTTPException(403, detail="Not allowed to edit environment variables")
    
    policy = admin_settings.get('api_key_policy', {})
    save_to_server = policy.get('save_user_keys_to_server', False)
    
    if not save_to_server:
        # 不保存到服务器，只返回成功（实际上是临时使用）
        return {"success": True, "saved_to_server": False}
    
    # 保存到服务器 .env 文件
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    try:
        for key, value in env_vars.items():
            if value:  # 只保存非空值
                set_key(env_path, key, value)
        return {"success": True, "saved_to_server": True}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to save env vars: {str(e)}")

@app.get("/api-key-policy", tags=["web"])
async def get_api_key_policy():
    """Get API key policy for users"""
    return admin_settings.get('api_key_policy', {
        'require_user_keys': False,
        'allow_server_keys': True,
        'save_user_keys_to_server': False,
    })

@contextmanager
def temp_env_vars(env_vars: dict):
    """
    临时设置环境变量的上下文管理器（线程安全版本）
    
    使用线程锁确保并发安全：
    1. 获取锁
    2. 保存原始环境变量
    3. 设置新的环境变量
    4. 清除翻译器缓存（强制重新创建）
    5. 执行翻译
    6. 恢复原始环境变量
    7. 再次清除缓存
    8. 释放锁
    
    Args:
        env_vars: 要临时设置的环境变量字典
    """
    if not env_vars:
        yield
        return
    
    # 使用线程锁确保同一时间只有一个请求在修改环境变量
    import threading
    if not hasattr(temp_env_vars, '_lock'):
        temp_env_vars._lock = threading.Lock()
    
    with temp_env_vars._lock:
        # 保存原始值
        original_values = {}
        for key in env_vars:
            original_values[key] = os.environ.get(key)
        
        # 清除翻译器缓存，强制重新创建（这样才能读取新的环境变量）
        from manga_translator.translators import translator_cache
        translator_cache.clear()
        
        try:
            # 设置新值
            for key, value in env_vars.items():
                if value:  # 只设置非空值
                    os.environ[key] = str(value)
            yield
        finally:
            # 恢复原始值
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            
            # 再次清除缓存，确保下次使用服务器的 Key 时重新创建
            translator_cache.clear()

async def apply_user_env_vars(user_env_vars_str: str, config: Config):
    """
    解析用户提供的环境变量，并检查策略
    
    Args:
        user_env_vars_str: JSON 字符串，包含用户的 API Keys
        config: 配置对象
    
    Returns:
        dict: 用户提供的环境变量字典，如果没有则返回 None
    """
    policy = admin_settings.get('api_key_policy', {})
    require_user_keys = policy.get('require_user_keys', False)
    allow_server_keys = policy.get('allow_server_keys', True)
    
    if not user_env_vars_str or user_env_vars_str.strip() in ('{}', ''):
        # 用户没有提供 API Keys
        
        if require_user_keys:
            # 强制要求用户提供 API Keys
            raise HTTPException(403, detail="User API keys are required")
        
        if not allow_server_keys:
            # 不允许使用服务器的 API Keys，但用户也没提供
            raise HTTPException(403, detail="Server API keys are not allowed, please provide your own")
        
        # 允许使用服务器的 API Keys（已经在环境变量中）
        return None
    
    try:
        user_env_vars = json.loads(user_env_vars_str)
        # 返回用户的环境变量，由调用者使用上下文管理器临时设置
        return {k: v for k, v in user_env_vars.items() if v and k.isupper()}
    except json.JSONDecodeError:
        return None

# --- i18n Endpoints ---

@app.get("/i18n/languages", tags=["web"])
async def get_languages():
    """Get available languages"""
    return get_available_locales()

@app.get("/i18n/{locale}", tags=["web"])
async def get_translation(locale: str):
    """Get translations for a specific locale"""
    return load_translation(locale)

# --- Admin Endpoints ---

@app.get("/admin/need-setup", tags=["admin"])
async def check_admin_setup():
    """检查是否需要首次设置管理员密码"""
    admin_password = admin_settings.get('admin_password')
    return {
        "need_setup": not admin_password or admin_password == ''
    }

@app.post("/admin/setup", tags=["admin"])
async def setup_admin_password(password: str = Form(...)):
    """首次设置管理员密码"""
    # 只有在没有密码时才允许设置
    if admin_settings.get('admin_password'):
        raise HTTPException(403, detail="Admin password already set")
    
    if not password or len(password) < 6:
        raise HTTPException(400, detail="Password must be at least 6 characters")
    
    # 保存密码到 admin_settings
    admin_settings['admin_password'] = password
    save_admin_settings()
    
    # 同时保存到 server_config（运行时使用）
    server_config['admin_password'] = password
    
    # 生成 token
    token = secrets.token_hex(32)
    valid_admin_tokens.add(token)
    
    logger.info("管理员密码已设置")
    return {"success": True, "token": token}

@app.post("/admin/login", tags=["admin"])
async def admin_login(password: str = Form(...)):
    """Admin login"""
    # 从 admin_settings 读取密码（持久化的）
    admin_password = admin_settings.get('admin_password')
    
    # 如果没有设置密码，返回错误（应该先设置）
    if not admin_password:
        raise HTTPException(400, detail="Admin password not set. Please setup first.")
    
    if password == admin_password:
        token = secrets.token_hex(32)
        valid_admin_tokens.add(token)
        return {"success": True, "token": token}
    return {"success": False, "message": "Invalid password"}

@app.post("/admin/change-password", tags=["admin"])
async def change_admin_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """更改管理员密码"""
    # 验证 token
    if not token or token not in valid_admin_tokens:
        raise HTTPException(401, detail="Unauthorized")
    
    # 验证旧密码
    admin_password = admin_settings.get('admin_password')
    if old_password != admin_password:
        return {"success": False, "message": "旧密码错误"}
    
    # 验证新密码
    if not new_password or len(new_password) < 6:
        return {"success": False, "message": "新密码至少需要6位"}
    
    # 更新密码
    admin_settings['admin_password'] = new_password
    save_admin_settings(admin_settings)
    
    # 同时更新运行时配置
    server_config['admin_password'] = new_password
    
    # 清除所有旧的 token（强制重新登录）
    valid_admin_tokens.clear()
    
    logger.info("管理员密码已更改")
    return {"success": True, "message": "密码已更改，请重新登录"}

@app.post("/user/login", tags=["web"])
async def user_login(password: str = Form(...)):
    """User login"""
    user_access = admin_settings.get('user_access', {})
    
    # 如果不需要密码，直接允许访问
    if not user_access.get('require_password', False):
        return {"success": True, "message": "No password required"}
    
    # 验证密码
    if password == user_access.get('user_password', ''):
        return {"success": True, "message": "Login successful"}
    
    return {"success": False, "message": "Invalid password"}

@app.get("/user/access", tags=["web"])
async def get_user_access():
    """Check if user access requires password"""
    user_access = admin_settings.get('user_access', {})
    return {
        "require_password": user_access.get('require_password', False)
    }

@app.get("/admin/settings", tags=["admin"])
async def get_admin_settings(token: str = Header(alias="X-Admin-Token")):
    """Get admin settings"""
    # 简化版：实际应该验证 token
    return admin_settings

@app.post("/admin/settings", tags=["admin"])
async def update_admin_settings(settings: dict, token: str = Header(alias="X-Admin-Token")):
    """Update admin settings"""
    # 简化版：实际应该验证 token
    # 支持部分更新
    for key, value in settings.items():
        if key in admin_settings:
            if isinstance(admin_settings[key], dict) and isinstance(value, dict):
                admin_settings[key].update(value)
            else:
                admin_settings[key] = value
    
    # 保存到文件
    if save_admin_settings(admin_settings):
        return {"success": True, "message": "Settings saved to file"}
    else:
        return {"success": False, "message": "Failed to save settings to file"}

@app.post("/admin/settings/parameter-visibility", tags=["admin"])
async def update_parameter_visibility(data: dict, token: str = Header(alias="X-Admin-Token")):
    """Update parameter visibility settings (hide/readonly/default values)"""
    if 'hidden_keys' in data:
        admin_settings['hidden_keys'] = data['hidden_keys']
    if 'readonly_keys' in data:
        admin_settings['readonly_keys'] = data['readonly_keys']
    if 'default_values' in data:
        admin_settings['default_values'] = data['default_values']
    
    # 保存到文件
    if save_admin_settings(admin_settings):
        return {"success": True, "message": "Settings saved to file"}
    else:
        return {"success": False, "message": "Failed to save settings to file"}

@app.get("/admin/server-config", tags=["admin"])
async def get_server_config(token: str = Header(alias="X-Admin-Token")):
    """Get server configuration"""
    return {
        "max_concurrent_tasks": server_config.get('max_concurrent_tasks', 3),
        "use_gpu": server_config.get('use_gpu', False),
        "verbose": server_config.get('verbose', False),
        "admin_config_path": ADMIN_CONFIG_PATH,
        "admin_config_exists": os.path.exists(ADMIN_CONFIG_PATH),
    }

@app.post("/admin/server-config", tags=["admin"])
async def update_server_config(config: dict, token: str = Header(alias="X-Admin-Token")):
    """Update server configuration"""
    if 'max_concurrent_tasks' in config:
        old_value = server_config.get('max_concurrent_tasks', 3)
        new_value = config['max_concurrent_tasks']
        server_config['max_concurrent_tasks'] = new_value
        
        # 如果并发数改变，重新初始化信号量
        if old_value != new_value:
            init_semaphore()
            logger.info(f"并发数已更新: {old_value} -> {new_value}")
        
        # 保存到 admin_config.json（持久化）
        try:
            admin_settings['max_concurrent_tasks'] = new_value
            save_admin_settings()
            logger.info(f"并发数已保存到配置文件: {new_value}")
        except Exception as e:
            logger.error(f"保存并发数到配置文件失败: {e}")
    
    return {"success": True}

@app.get("/announcement", tags=["web"])
async def get_announcement():
    """获取公告（用户端）"""
    announcement = admin_settings.get('announcement', {})
    if announcement.get('enabled', False):
        return {
            "enabled": True,
            "message": announcement.get('message', ''),
            "type": announcement.get('type', 'info')
        }
    return {"enabled": False}

@app.post("/admin/announcement", tags=["admin"])
async def update_announcement(announcement: dict, token: str = Header(alias="X-Admin-Token")):
    """更新公告（管理员）"""
    admin_settings['announcement'] = announcement
    save_admin_settings()
    logger.info(f"公告已更新: enabled={announcement.get('enabled')}, type={announcement.get('type')}")
    return {"success": True}

# --- File Upload Endpoints ---

@app.post("/upload/font", tags=["admin"])
async def upload_font(file: UploadFile = File(...), token: str = Header(alias="X-Admin-Token", default=None)):
    """Upload a font file"""
    # 检查权限
    if not admin_settings.get('permissions', {}).get('can_upload_fonts', True):
        raise HTTPException(403, detail="Font upload is disabled")
    
    if not file.filename.lower().endswith(('.ttf', '.otf', '.ttc')):
        raise HTTPException(400, detail="Invalid font file format")
    
    from manga_translator.utils import BASE_PATH
    fonts_dir = os.path.join(BASE_PATH, 'fonts')
    os.makedirs(fonts_dir, exist_ok=True)
    
    file_path = os.path.join(fonts_dir, file.filename)
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    return {"success": True, "filename": file.filename}

@app.post("/upload/prompt", tags=["admin"])
async def upload_prompt(file: UploadFile = File(...), token: str = Header(alias="X-Admin-Token", default=None)):
    """Upload a high-quality translation prompt file"""
    # 检查权限
    if not admin_settings.get('permissions', {}).get('can_upload_prompts', True):
        raise HTTPException(403, detail="Prompt upload is disabled")
    
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(400, detail="Invalid prompt file format (must be .json)")
    
    # 禁止上传系统提示词文件名
    if file.filename in ['system_prompt_hq.json', 'system_prompt_line_break.json']:
        raise HTTPException(403, detail="Cannot overwrite system prompt files")
    
    from manga_translator.utils import BASE_PATH
    dict_dir = os.path.join(BASE_PATH, 'dict')
    os.makedirs(dict_dir, exist_ok=True)
    
    file_path = os.path.join(dict_dir, file.filename)
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    return {"success": True, "filename": file.filename}

@app.get("/prompts", tags=["web"])
async def list_prompts():
    """List available prompt files (excluding system prompts)"""
    try:
        from manga_translator.utils import BASE_PATH
        dict_dir = os.path.join(BASE_PATH, 'dict')
        prompts = []
        
        print(f"[DEBUG] Listing prompts from: {dict_dir}")
        print(f"[DEBUG] dict_dir exists: {os.path.exists(dict_dir)}")
        
        # 从 dict 目录读取（桌面版使用的目录）
        if os.path.exists(dict_dir):
            files = os.listdir(dict_dir)
            print(f"[DEBUG] Found {len(files)} files in dict_dir")
            for f in files:
                # 过滤掉系统提示词文件
                if f.lower().endswith('.json') and f not in [
                    'system_prompt_hq.json',
                    'system_prompt_line_break.json'
                ]:
                    prompts.append(f)
                    print(f"[DEBUG] Added prompt: {f}")
        
        print(f"[DEBUG] Returning {len(prompts)} prompts")
        return sorted(prompts)
    except Exception as e:
        print(f"[ERROR] Failed to list prompts: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=f"Failed to list prompts: {str(e)}")

@app.get("/prompts/{filename}", tags=["admin"])
async def get_prompt(filename: str, token: str = Header(alias="X-Admin-Token", default=None)):
    """Get prompt file content (admin only)"""
    from manga_translator.utils import BASE_PATH
    dict_dir = os.path.join(BASE_PATH, 'dict')
    
    # 在 dict 目录查找
    file_path = os.path.join(dict_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Prompt file not found")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {"filename": filename, "content": content}

@app.delete("/prompts/{filename}", tags=["admin"])
async def delete_prompt(filename: str, token: str = Header(alias="X-Admin-Token")):
    """Delete a prompt file (admin only)"""
    # 检查权限
    if not admin_settings.get('permissions', {}).get('can_delete_prompts', True):
        raise HTTPException(403, detail="Prompt deletion is disabled")
    
    from manga_translator.utils import BASE_PATH
    dict_dir = os.path.join(BASE_PATH, 'dict')
    
    # 禁止删除系统提示词
    if filename in ['system_prompt_hq.json', 'system_prompt_line_break.json']:
        raise HTTPException(403, detail="Cannot delete system prompt files")
    
    # 在 dict 目录查找
    file_path = os.path.join(dict_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Prompt file not found")
    
    try:
        os.remove(file_path)
        return {"success": True, "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to delete file: {str(e)}")

@app.delete("/fonts/{filename}", tags=["admin"])
async def delete_font(filename: str, token: str = Header(alias="X-Admin-Token")):
    """Delete a font file (admin only)"""
    # 检查权限
    if not admin_settings.get('permissions', {}).get('can_delete_fonts', True):
        raise HTTPException(403, detail="Font deletion is disabled")
    
    from manga_translator.utils import BASE_PATH
    fonts_dir = os.path.join(BASE_PATH, 'fonts')
    
    # 在 fonts 目录查找
    file_path = os.path.join(fonts_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Font file not found")
    
    try:
        os.remove(file_path)
        return {"success": True, "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to delete file: {str(e)}")

# --- Log Endpoints ---
from collections import deque, defaultdict
from datetime import datetime
import logging
import threading
import uuid

# 基于任务ID的日志队列（每个任务有独立的日志队列）
task_logs = defaultdict(lambda: deque(maxlen=500))  # 每个任务最多保存500条日志
task_logs_lock = threading.Lock()

# 全局日志队列（用于管理员查看所有日志）
global_log_queue = deque(maxlen=1000)

# 当前任务ID的线程本地存储
import contextvars
current_task_id = contextvars.ContextVar('current_task_id', default=None)

def generate_task_id():
    """生成唯一的任务ID"""
    return str(uuid.uuid4())

def set_task_id(task_id: str):
    """设置当前任务ID"""
    current_task_id.set(task_id)

def get_task_id():
    """获取当前任务ID"""
    return current_task_id.get()

def add_log(message: str, level: str = "INFO", task_id: str = None):
    """添加日志到队列（支持任务隔离）"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    
    # 如果没有指定task_id，尝试从上下文获取
    if task_id is None:
        task_id = get_task_id()
    
    with task_logs_lock:
        # 添加到全局日志队列
        global_log_queue.append(log_entry)
        
        # 如果有task_id，也添加到任务专属日志队列
        if task_id:
            log_entry_with_id = log_entry.copy()
            log_entry_with_id['task_id'] = task_id
            task_logs[task_id].append(log_entry_with_id)
    
    print(f"[{level}] [{task_id[:8] if task_id else 'GLOBAL'}] {message}")  # 同时输出到控制台

# 创建自定义日志处理器，捕获manga_translator的日志
class WebLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # 提取日志级别和消息
            level = record.levelname
            # 从上下文获取task_id
            task_id = get_task_id()
            add_log(msg, level, task_id)
        except Exception:
            self.handleError(record)

# 设置日志处理器
web_log_handler = WebLogHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
web_log_handler.setFormatter(formatter)

# 只添加到根logger以捕获所有模块的日志（包括OCR、检测器等）
root_logger = logging.getLogger()
root_logger.addHandler(web_log_handler)
if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
    root_logger.setLevel(logging.INFO)

@app.get("/logs", tags=["web"])
async def get_logs(level: str = None, limit: int = 100, task_id: str = None):
    """
    获取日志
    - level: 日志级别过滤（INFO, WARNING, ERROR等）
    - limit: 返回的日志数量限制
    - task_id: 任务ID（如果指定，只返回该任务的日志；否则返回全局日志）
    """
    with task_logs_lock:
        if task_id:
            # 返回指定任务的日志
            logs = list(task_logs.get(task_id, []))
        else:
            # 返回全局日志
            logs = list(global_log_queue)
    
    # 按级别过滤
    if level:
        logs = [log for log in logs if log['level'].lower() == level.lower()]
    
    # 限制数量（返回最新的）
    if len(logs) > limit:
        logs = logs[-limit:]
    
    return logs

@app.get("/admin/logs/export", tags=["admin"])
async def export_logs(token: str = Header(alias="X-Admin-Token"), task_id: str = None):
    """导出日志为文本文件"""
    with task_logs_lock:
        if task_id:
            logs = list(task_logs.get(task_id, []))
            filename = f"logs_{task_id[:8]}.txt"
        else:
            logs = list(global_log_queue)
            filename = f"logs_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # 生成日志文本
    log_text = "\n".join([
        f"[{log['timestamp']}] [{log['level']}] {log['message']}"
        for log in logs
    ])
    
    return StreamingResponse(
        io.BytesIO(log_text.encode('utf-8')),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# 活动任务跟踪
active_tasks = {}  # task_id -> {"start_time": ..., "status": ..., "cancel_requested": False, "task": asyncio.Task}
active_tasks_lock = threading.Lock()

@app.get("/admin/tasks", tags=["admin"])
async def get_active_tasks(token: str = Header(alias="X-Admin-Token")):
    """获取所有活动任务"""
    with active_tasks_lock:
        tasks = []
        for task_id, info in active_tasks.items():
            tasks.append({
                "task_id": task_id,
                "start_time": info["start_time"],
                "status": info["status"],
                "duration": (datetime.now() - datetime.fromisoformat(info["start_time"])).total_seconds()
            })
        return tasks

@app.post("/admin/tasks/{task_id}/cancel", tags=["admin"])
async def cancel_task(task_id: str, force: bool = False, token: str = Header(alias="X-Admin-Token")):
    """
    取消指定的翻译任务
    
    Args:
        task_id: 任务ID
        force: 是否强制取消（立即终止任务，不等待检查点）
        token: 管理员令牌
    """
    with active_tasks_lock:
        if task_id in active_tasks:
            active_tasks[task_id]["cancel_requested"] = True
            
            if force:
                # 强制取消：直接调用 asyncio.Task.cancel()
                task = active_tasks[task_id].get("task")
                if task and not task.done():
                    task.cancel()
                    add_log(f"管理员强制取消任务: {task_id[:8]}", "WARNING")
                    return {"success": True, "message": "任务已强制终止"}
                else:
                    add_log(f"管理员请求强制取消任务，但任务已完成: {task_id[:8]}", "INFO")
                    return {"success": True, "message": "任务已完成，无需取消"}
            else:
                # 协作式取消：设置标志，等待任务在检查点响应
                add_log(f"管理员请求取消任务: {task_id[:8]}", "WARNING")
                return {"success": True, "message": "取消请求已发送（协作式取消）"}
        else:
            raise HTTPException(404, detail="任务不存在或已完成")

def register_active_task(task_id: str, task: asyncio.Task = None):
    """注册活动任务"""
    with active_tasks_lock:
        active_tasks[task_id] = {
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "cancel_requested": False,
            "task": task  # 保存 asyncio.Task 引用以便强制取消
        }

def unregister_active_task(task_id: str):
    """注销活动任务"""
    with active_tasks_lock:
        if task_id in active_tasks:
            del active_tasks[task_id]

def is_task_cancelled(task_id: str) -> bool:
    """检查任务是否被取消"""
    with active_tasks_lock:
        if task_id in active_tasks:
            return active_tasks[task_id].get("cancel_requested", False)
        return False



@app.post("/register", response_description="no response", tags=["internal-api"])
async def register_instance(instance: ExecutorInstance, req: Request, req_nonce: str = Header(alias="X-Nonce")):
    if req_nonce != nonce:
        raise HTTPException(401, detail="Invalid nonce")
    instance.ip = req.client.host
    executor_instances.register(instance)

def transform_to_image(ctx):
    # 检查 ctx.result 是否存在
    if ctx.result is None:
        raise HTTPException(500, detail="Translation failed: no result image generated")
    
    # 检查是否使用占位符（在web模式下final.png保存后会设置此标记）
    if hasattr(ctx, 'use_placeholder') and ctx.use_placeholder:
        # ctx.result已经是1x1占位符图片，快速传输
        img_byte_arr = io.BytesIO()
        ctx.result.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    # 返回完整的翻译结果
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

def transform_to_json(ctx):
    return to_translation(ctx).model_dump_json().encode("utf-8")

def transform_to_bytes(ctx):
    return to_translation(ctx).to_bytes()

@app.post("/translate/json", response_model=TranslationResponse, tags=["api", "json"],response_description="json strucure inspired by the ichigo translator extension")
async def translate_json(req: Request, data: TranslateRequest):
    ctx = await get_ctx(req, data.config, data.image, "save_json")
    return to_translation(ctx)

@app.post("/translate/bytes", response_class=StreamingResponse, tags=["api", "json"],response_description="custom byte structure for decoding look at examples in 'examples/response.*'")
async def bytes(req: Request, data: TranslateRequest):
    ctx = await get_ctx(req, data.config, data.image, "save_json")
    return StreamingResponse(content=to_translation(ctx).to_bytes())

@app.post("/translate/image", response_description="the result image", tags=["api", "json"],response_class=StreamingResponse)
async def image(req: Request, data: TranslateRequest) -> StreamingResponse:
    ctx = await get_ctx(req, data.config, data.image, "normal")
    
    if not ctx.result:
        raise HTTPException(500, detail="Translation failed: no result image generated")
    
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/translate/json/stream", response_class=StreamingResponse,tags=["api", "json"], response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_json(req: Request, data: TranslateRequest) -> StreamingResponse:
    return await while_streaming(req, transform_to_json, data.config, data.image, "save_json")

@app.post("/translate/bytes/stream", response_class=StreamingResponse, tags=["api", "json"],response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_bytes(req: Request, data: TranslateRequest)-> StreamingResponse:
    return await while_streaming(req, transform_to_bytes,data.config, data.image, "save_json")

@app.post("/translate/image/stream", response_class=StreamingResponse, tags=["api", "json"], response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_image(req: Request, data: TranslateRequest) -> StreamingResponse:
    return await while_streaming(req, transform_to_image, data.config, data.image, "normal")

@app.post("/translate/with-form/json", response_model=TranslationResponse, tags=["api", "form"],response_description="json strucure inspired by the ichigo translator extension")
async def translate_json_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    img = await image.read()
    conf = parse_config(config)
    ctx = await get_ctx(req, conf, img, "save_json")
    return to_translation(ctx)

@app.post("/translate/with-form/bytes", response_class=StreamingResponse, tags=["api", "form"],response_description="custom byte structure for decoding look at examples in 'examples/response.*'")
async def bytes_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    img = await image.read()
    conf = parse_config(config)
    ctx = await get_ctx(req, conf, img, "save_json")
    return StreamingResponse(content=to_translation(ctx).to_bytes())

@app.post("/translate/with-form/image", response_description="the result image", tags=["api", "form"],response_class=StreamingResponse)
async def image_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")) -> StreamingResponse:
    img = await image.read()
    conf = parse_config(config)
    ctx = await get_ctx(req, conf, img, "normal")
    
    if not ctx.result:
        raise HTTPException(500, detail="Translation failed: no result image generated")
    
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/translate/with-form/json/stream", response_class=StreamingResponse, tags=["api", "form"],response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_json_form(req: Request, image: UploadFile = File(...), config: str = Form("{}")) -> StreamingResponse:
    img = await image.read()
    conf = parse_config(config)
    # 标记这是Web前端调用，用于占位符优化
    conf._is_web_frontend = True
    return await while_streaming(req, transform_to_json, conf, img, "save_json")



@app.post("/translate/with-form/bytes/stream", response_class=StreamingResponse,tags=["api", "form"], response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_bytes_form(req: Request, image: UploadFile = File(...), config: str = Form("{}"))-> StreamingResponse:
    img = await image.read()
    conf = parse_config(config)
    return await while_streaming(req, transform_to_bytes, conf, img, "save_json")

@app.post("/translate/with-form/image/stream", response_class=StreamingResponse, tags=["api", "form"], response_description="Standard streaming endpoint - returns complete image data. Suitable for API calls and scripts.")
async def stream_image_form(req: Request, image: UploadFile = File(...), config: str = Form("{}"), user_env_vars: str = Form("{}")) -> StreamingResponse:
    """通用流式端点：返回完整图片数据，适用于API调用和comicread脚本"""
    img = await image.read()
    conf = parse_config(config)
    
    # 解析用户提供的 API Keys并存储到config中
    env_vars = await apply_user_env_vars(user_env_vars, conf)
    conf._user_env_vars = env_vars  # 存储到config对象中，由翻译器使用
    
    # 标记为通用模式，不使用占位符优化
    conf._web_frontend_optimized = False
    return await while_streaming(req, transform_to_image, conf, img, "normal")

@app.post("/translate/with-form/image/stream/web", response_class=StreamingResponse, tags=["api", "form"], response_description="Web frontend optimized streaming endpoint - uses placeholder optimization for faster response.")
async def stream_image_form_web(req: Request, image: UploadFile = File(...), config: str = Form("{}")) -> StreamingResponse:
    """Web前端专用端点：使用占位符优化，提供极速体验"""
    img = await image.read()
    conf = parse_config(config)
    # 标记为Web前端优化模式，使用占位符优化
    conf._web_frontend_optimized = True
    return await while_streaming(req, transform_to_image, conf, img, "normal")

@app.post("/queue-size", response_model=int, tags=["api", "json"])
async def queue_size() -> int:
    return len(task_queue.queue)


@app.api_route("/result/{folder_name}/final.png", methods=["GET", "HEAD"], tags=["api", "file"])
async def get_result_by_folder(folder_name: str):
    """根据文件夹名称获取翻译结果图片"""
    result_dir = "../result"
    if not os.path.exists(result_dir):
        raise HTTPException(404, detail="Result directory not found")

    folder_path = os.path.join(result_dir, folder_name)
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(404, detail=f"Folder {folder_name} not found")

    final_png_path = os.path.join(folder_path, "final.png")
    if not os.path.exists(final_png_path):
        raise HTTPException(404, detail="final.png not found in folder")

    async def file_iterator():
        with open(final_png_path, "rb") as f:
            yield f.read()

    return StreamingResponse(
        file_iterator(),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=final.png"}
    )

@app.post("/translate/batch/json", response_model=list[TranslationResponse], tags=["api", "json", "batch"])
async def translate_batch_json(req: Request, data: BatchTranslateRequest):
    """Batch translate images and return JSON format results"""
    results = await get_batch_ctx(req, data.config, data.images, data.batch_size, "normal")
    return [to_translation(ctx) for ctx in results]

@app.post("/translate/batch/images", response_description="Zip file containing translated images", tags=["api", "batch"])
async def batch_images(req: Request, data: BatchTranslateRequest):
    """Batch translate images and return zip archive containing translated images"""
    import zipfile
    import tempfile
    
    try:
        add_log(f"批量翻译请求: {len(data.images)} 张图片, batch_size={data.batch_size}", "INFO")
        
        # 如果config是字典，转换为Config对象
        if isinstance(data.config, dict):
            config = Config.parse_obj(data.config)
        else:
            config = data.config
        
        results = await get_batch_ctx(req, config, data.images, data.batch_size, "normal")
        add_log(f"批量翻译完成: 收到 {len(results)} 个结果", "INFO")
            
    except Exception as e:
        add_log(f"批量翻译失败: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=f"Batch translation failed: {str(e)}")
    
    # Create temporary ZIP file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        tmp_file_name = tmp_file.name
    
    # 文件句柄已关闭，现在可以安全地写入
    image_count = 0
    with zipfile.ZipFile(tmp_file_name, 'w') as zip_file:
        for i, ctx in enumerate(results):
            if ctx and ctx.result:
                img_byte_arr = io.BytesIO()
                ctx.result.save(img_byte_arr, format="PNG")
                zip_file.writestr(f"translated_{i+1}.png", img_byte_arr.getvalue())
                image_count += 1
    
    add_log(f"ZIP文件创建完成: 包含 {image_count} 张图片", "INFO")
    
    # 读取 ZIP 文件内容
    with open(tmp_file_name, 'rb') as f:
        zip_data = f.read()
    
    # 清理临时文件（Windows 兼容）
    try:
        os.unlink(tmp_file_name)
    except PermissionError:
        # Windows 上可能需要延迟删除
        import atexit
        atexit.register(lambda: os.unlink(tmp_file_name) if os.path.exists(tmp_file_name) else None)
    
    return StreamingResponse(
        io.BytesIO(zip_data),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=translated_images.zip"}
        )

@app.post("/translate/export/original", response_class=StreamingResponse, tags=["api", "export"])
async def export_original(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """导出原文（ZIP 压缩包：JSON + TXT）"""
    workflow = "export_original"
    import json
    import tempfile
    import zipfile
    
    img = await image.read()
    conf = parse_config(config)
    
    # 使用指定的 workflow 进行处理
    ctx = await get_ctx(req, conf, img, workflow)
    
    # 将结果转换为 JSON 格式
    translation_data = to_translation(ctx)
    
    # 创建临时 JSON 文件（使用与主翻译程序相同的格式）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_json:
        # 将 Pydantic 模型转换为字典
        response_dict = translation_data.model_dump()
        
        # 构建与主翻译程序相同的 JSON 结构
        json_data = {
            "temp_image": response_dict
        }
        json.dump(json_data, tmp_json, ensure_ascii=False, indent=4)
        tmp_json_path = tmp_json.name
    
    try:
        # 生成 TXT 文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_txt:
            tmp_txt_path = tmp_txt.name
        
        # 直接导入模块，避免触发 __init__.py 中的其他导入
        import sys
        import os
        workflow_service_path = os.path.join(os.path.dirname(__file__), '..', '..', 'desktop_qt_ui', 'services', 'workflow_service.py')
        workflow_service_path = os.path.abspath(workflow_service_path)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("workflow_service", workflow_service_path)
        workflow_service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow_service)
        
        # 获取默认模板
        template_path = workflow_service.ensure_default_template_exists()
        if not template_path:
            raise HTTPException(500, detail="无法创建或找到默认模板文件")
        
        ui_generate_original_text = workflow_service.generate_original_text
        txt_path = ui_generate_original_text(tmp_json_path, template_path=template_path, output_path=tmp_txt_path)
        
        if txt_path.startswith("Error"):
            raise HTTPException(500, detail=txt_path)
        
        # 创建 ZIP 文件
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 添加 JSON 文件
            with open(tmp_json_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            zip_file.writestr("translation.json", json_content)
            
            # 添加 TXT 文件
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_content = f.read()
            zip_file.writestr("original.txt", txt_content)
        
        # 清理临时文件
        os.unlink(tmp_json_path)
        os.unlink(txt_path)
        
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=original_export.zip"}
        )
    except Exception as e:
        # 清理临时文件
        if os.path.exists(tmp_json_path):
            os.unlink(tmp_json_path)
        if 'txt_path' in locals() and os.path.exists(txt_path):
            os.unlink(txt_path)
        raise HTTPException(500, detail=f"Error exporting files: {str(e)}")

@app.post("/translate/export/translated", response_class=StreamingResponse, tags=["api", "export"])
async def export_translated(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """导出译文（ZIP 压缩包：JSON + TXT）"""
    workflow = "save_json"
    import json
    import tempfile
    import zipfile
    
    img = await image.read()
    conf = parse_config(config)
    
    # 使用指定的 workflow 进行处理
    ctx = await get_ctx(req, conf, img, workflow)
    
    # 将结果转换为 JSON 格式
    translation_data = to_translation(ctx)
    
    # 创建临时 JSON 文件（使用与主翻译程序相同的格式）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_json:
        # 将 Pydantic 模型转换为字典
        response_dict = translation_data.model_dump()
        
        # 构建与主翻译程序相同的 JSON 结构
        json_data = {
            "temp_image": response_dict
        }
        json.dump(json_data, tmp_json, ensure_ascii=False, indent=4)
        tmp_json_path = tmp_json.name
    
    try:
        # 生成 TXT 文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_txt:
            tmp_txt_path = tmp_txt.name
        
        # 直接导入模块，避免触发 __init__.py 中的其他导入
        import sys
        import os
        workflow_service_path = os.path.join(os.path.dirname(__file__), '..', '..', 'desktop_qt_ui', 'services', 'workflow_service.py')
        workflow_service_path = os.path.abspath(workflow_service_path)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("workflow_service", workflow_service_path)
        workflow_service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow_service)
        
        # 获取默认模板
        template_path = workflow_service.ensure_default_template_exists()
        if not template_path:
            raise HTTPException(500, detail="无法创建或找到默认模板文件")
        
        ui_generate_translated_text = workflow_service.generate_translated_text
        txt_path = ui_generate_translated_text(tmp_json_path, template_path=template_path, output_path=tmp_txt_path)
        
        if txt_path.startswith("Error"):
            raise HTTPException(500, detail=txt_path)
        
        # 创建 ZIP 文件
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 添加 JSON 文件
            with open(tmp_json_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            zip_file.writestr("translation.json", json_content)
            
            # 添加 TXT 文件
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_content = f.read()
            zip_file.writestr("translated.txt", txt_content)
        
        # 清理临时文件
        os.unlink(tmp_json_path)
        os.unlink(txt_path)
        
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=translated_export.zip"}
        )
    except Exception as e:
        # 清理临时文件
        if os.path.exists(tmp_json_path):
            os.unlink(tmp_json_path)
        if 'txt_path' in locals() and os.path.exists(txt_path):
            os.unlink(txt_path)
        raise HTTPException(500, detail=f"Error exporting files: {str(e)}")

@app.post("/translate/upscale", response_class=StreamingResponse, tags=["api", "process"])
async def upscale_only(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """仅超分（图片超分辨率）"""
    img = await image.read()
    conf = parse_config(config)
    ctx = await get_ctx(req, conf, img, "upscale_only")
    
    if ctx.result:
        img_byte_arr = io.BytesIO()
        ctx.result.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")
    else:
        raise HTTPException(500, detail="Upscaling failed")

@app.post("/translate/colorize", response_class=StreamingResponse, tags=["api", "process"])
async def colorize_only(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """仅上色（黑白图片上色）"""
    img = await image.read()
    conf = parse_config(config)
    ctx = await get_ctx(req, conf, img, "colorize_only")
    
    if ctx.result:
        img_byte_arr = io.BytesIO()
        ctx.result.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")
    else:
        raise HTTPException(500, detail="Colorization failed")

@app.post("/translate/inpaint", response_class=StreamingResponse, tags=["api", "process"])
async def inpaint_only(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """仅修复（检测文字并修复图片）"""
    img = await image.read()
    conf = parse_config(config)
    ctx = await get_ctx(req, conf, img, "inpaint_only")
    
    if ctx.result:
        img_byte_arr = io.BytesIO()
        ctx.result.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")
    else:
        raise HTTPException(500, detail="Inpainting failed")

@app.post("/translate/import/json", response_class=StreamingResponse, tags=["api", "import"])
async def import_json_and_render(req: Request, image: UploadFile = File(...), json_file: UploadFile = File(...), config: str = Form("{}")):
    """导入 JSON + 图片，返回渲染后的图片（load_text workflow）"""
    import json
    import tempfile
    from manga_translator.utils.path_manager import get_work_dir
    from PIL import Image as PILImage
    
    img_bytes = await image.read()
    json_content = await json_file.read()
    conf = parse_config(config)
    
    # 使用临时文件名
    temp_name = f"temp_{secrets.token_hex(8)}"
    # 先创建临时图片路径（在当前目录或result目录）
    temp_image_path = os.path.join("result", f"{temp_name}.png")
    os.makedirs("result", exist_ok=True)
    
    # 保存 JSON 到临时文件
    work_dir = get_work_dir(temp_image_path)
    json_dir = os.path.join(work_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)
    
    json_path = os.path.join(json_dir, f"{temp_name}_translations.json")
    
    try:
        # 写入 JSON 文件
        with open(json_path, 'wb') as f:
            f.write(json_content)
        
        # 保存图片到临时位置（使用相同的名称）
        temp_image = PILImage.open(io.BytesIO(img_bytes))
        temp_image.save(temp_image_path)
        
        # 重新加载图片并设置 name 属性
        temp_image = PILImage.open(temp_image_path)
        temp_image.name = temp_image_path
        
        # 使用 load_text workflow，通过 get_ctx 调用（支持任务队列）
        ctx = await get_ctx(req, conf, temp_image, "load_text")
        
        if ctx.result:
            img_byte_arr = io.BytesIO()
            ctx.result.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            
            # 清理临时文件
            if os.path.exists(json_path):
                os.unlink(json_path)
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            
            return StreamingResponse(img_byte_arr, media_type="image/png")
        else:
            # 清理临时文件
            if os.path.exists(json_path):
                os.unlink(json_path)
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise HTTPException(500, detail="Failed to render image")
    
    except Exception as e:
        # 清理临时文件
        if os.path.exists(json_path):
            os.unlink(json_path)
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        raise HTTPException(500, detail=f"Error importing and rendering: {str(e)}")

@app.post("/translate/import/txt", response_class=StreamingResponse, tags=["api", "import"])
async def import_txt_and_render(req: Request, image: UploadFile = File(...), txt_file: UploadFile = File(...), json_file: UploadFile = File(...), config: str = Form("{}"), template: UploadFile = File(None)):
    """导入 TXT + JSON + 图片，返回渲染后的图片（使用 UI 层的导入逻辑）"""
    import tempfile
    import importlib.util
    from manga_translator.utils.path_manager import get_work_dir
    from PIL import Image as PILImage
    
    # 直接导入 workflow_service 模块，避免触发 __init__.py
    workflow_service_path = os.path.join(os.path.dirname(__file__), '..', '..', 'desktop_qt_ui', 'services', 'workflow_service.py')
    workflow_service_path = os.path.abspath(workflow_service_path)
    spec = importlib.util.spec_from_file_location("workflow_service", workflow_service_path)
    workflow_service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow_service)
    
    safe_update_large_json_from_text = workflow_service.safe_update_large_json_from_text
    ensure_default_template_exists = workflow_service.ensure_default_template_exists
    
    img_bytes = await image.read()
    txt_content = await txt_file.read()
    json_content = await json_file.read()
    conf = parse_config(config)
    
    # 使用临时文件名
    temp_name = f"temp_{secrets.token_hex(8)}"
    # 先创建临时图片路径（在result目录）
    temp_image_path = os.path.join("result", f"{temp_name}.png")
    os.makedirs("result", exist_ok=True)
    
    # 创建工作目录
    work_dir = get_work_dir(temp_image_path)
    json_dir = os.path.join(work_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)
    
    json_path = os.path.join(json_dir, f"{temp_name}_translations.json")
    temp_txt_path = os.path.join(work_dir, f"{temp_name}_temp.txt")
    
    try:
        # 保存原始 JSON 文件
        with open(json_path, 'wb') as f:
            f.write(json_content)
        
        # 保存 TXT 文件
        with open(temp_txt_path, 'wb') as f:
            f.write(txt_content)
        
        # 获取模板路径
        if template:
            # 如果用户提供了模板，保存它
            template_content = await template.read()
            temp_template_path = os.path.join(work_dir, f"{temp_name}_template.txt")
            with open(temp_template_path, 'wb') as f:
                f.write(template_content)
            template_path = temp_template_path
        else:
            # 使用默认模板
            template_path = ensure_default_template_exists()
            if not template_path:
                raise HTTPException(500, detail="无法找到或创建默认模板文件")
        
        # 使用 UI 层的导入逻辑（支持模板解析和模糊匹配）
        import_result = safe_update_large_json_from_text(temp_txt_path, json_path, template_path)
        
        if import_result.startswith("错误"):
            raise HTTPException(400, detail=import_result)
        
        # 保存图片到临时位置
        temp_image = PILImage.open(io.BytesIO(img_bytes))
        temp_image.save(temp_image_path)
        
        # 重新加载图片并设置 name 属性
        temp_image = PILImage.open(temp_image_path)
        temp_image.name = temp_image_path
        
        # 使用 load_text workflow，通过 get_ctx 调用（支持任务队列）
        ctx = await get_ctx(req, conf, temp_image, "load_text")
        
        if ctx.result:
            img_byte_arr = io.BytesIO()
            ctx.result.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            
            # 清理临时文件
            if os.path.exists(json_path):
                os.unlink(json_path)
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            
            return StreamingResponse(img_byte_arr, media_type="image/png")
        else:
            # 清理临时文件
            if os.path.exists(json_path):
                os.unlink(json_path)
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise HTTPException(500, detail="Failed to render image")
    
    except Exception as e:
        # 清理临时文件
        if os.path.exists(json_path):
            os.unlink(json_path)
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        raise HTTPException(500, detail=f"Error importing and rendering: {str(e)}")

@app.post("/translate/import/json/stream", response_class=StreamingResponse, tags=["api", "import", "stream"])
async def import_json_and_render_stream(req: Request, image: UploadFile = File(...), json_file: UploadFile = File(...), config: str = Form("{}")):
    """导入 JSON + 图片，返回渲染后的图片（流式，支持进度）"""
    import json
    from manga_translator.utils.path_manager import get_work_dir
    from PIL import Image
    
    img = await image.read()
    json_content = await json_file.read()
    conf = parse_config(config)
    
    # 使用临时文件名
    temp_name = f"temp_{secrets.token_hex(8)}"
    # 先创建临时图片路径（在result目录）
    temp_image_path = os.path.join("result", f"{temp_name}.png")
    os.makedirs("result", exist_ok=True)
    
    # 保存 JSON 到临时文件
    work_dir = get_work_dir(temp_image_path)
    json_dir = os.path.join(work_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)
    
    json_path = os.path.join(json_dir, f"{temp_name}_translations.json")
    
    try:
        # 写入 JSON 文件
        with open(json_path, 'wb') as f:
            f.write(json_content)
        
        # 保存图片到临时位置
        temp_image = Image.open(io.BytesIO(img))
        temp_image.save(temp_image_path)
        
        # 重新加载图片并设置 name 属性
        temp_image = Image.open(temp_image_path)
        temp_image.name = temp_image_path
        
        # 使用流式翻译，传递 PIL Image 对象
        # 注意：流式响应中不能在 finally 中删除文件，因为响应还在进行中
        # 临时文件会在 result 目录中累积，需要定期清理
        return await while_streaming(req, transform_to_image, conf, temp_image, "load_text")
    
    except Exception as e:
        # 只在出错时清理临时文件
        try:
            if os.path.exists(json_path):
                os.unlink(json_path)
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
        except:
            pass  # 忽略清理错误
        raise

@app.post("/translate/import/txt/stream", response_class=StreamingResponse, tags=["api", "import", "stream"])
async def import_txt_and_render_stream(req: Request, image: UploadFile = File(...), txt_file: UploadFile = File(...), json_file: UploadFile = File(...), config: str = Form("{}"), template: UploadFile = File(None)):
    """导入 TXT + JSON + 图片，返回渲染后的图片（流式，支持进度，使用 UI 层的导入逻辑）"""
    import tempfile
    import importlib.util
    from manga_translator.utils.path_manager import get_work_dir
    from PIL import Image
    
    # 直接导入 workflow_service 模块，避免触发 __init__.py
    workflow_service_path = os.path.join(os.path.dirname(__file__), '..', '..', 'desktop_qt_ui', 'services', 'workflow_service.py')
    workflow_service_path = os.path.abspath(workflow_service_path)
    spec = importlib.util.spec_from_file_location("workflow_service", workflow_service_path)
    workflow_service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow_service)
    
    safe_update_large_json_from_text = workflow_service.safe_update_large_json_from_text
    ensure_default_template_exists = workflow_service.ensure_default_template_exists
    
    img = await image.read()
    txt_content = await txt_file.read()
    json_content = await json_file.read()
    conf = parse_config(config)
    
    # 使用临时文件名
    temp_name = f"temp_{secrets.token_hex(8)}"
    # 先创建临时图片路径（在result目录）
    temp_image_path = os.path.join("result", f"{temp_name}.png")
    os.makedirs("result", exist_ok=True)
    
    # 创建工作目录
    work_dir = get_work_dir(temp_image_path)
    json_dir = os.path.join(work_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)
    
    json_path = os.path.join(json_dir, f"{temp_name}_translations.json")
    temp_txt_path = os.path.join(work_dir, f"{temp_name}_temp.txt")
    
    try:
        # 保存原始 JSON 文件
        with open(json_path, 'wb') as f:
            f.write(json_content)
        
        # 保存 TXT 文件
        with open(temp_txt_path, 'wb') as f:
            f.write(txt_content)
        
        # 获取模板路径
        if template:
            # 如果用户提供了模板，保存它
            template_content = await template.read()
            temp_template_path = os.path.join(work_dir, f"{temp_name}_template.txt")
            with open(temp_template_path, 'wb') as f:
                f.write(template_content)
            template_path = temp_template_path
        else:
            # 使用默认模板
            template_path = ensure_default_template_exists()
            if not template_path:
                raise HTTPException(500, detail="无法找到或创建默认模板文件")
        
        # 使用 UI 层的导入逻辑
        import_result = safe_update_large_json_from_text(temp_txt_path, json_path, template_path)
        
        if import_result.startswith("错误"):
            raise HTTPException(400, detail=import_result)
        
        # 保存图片到临时位置
        temp_image = Image.open(io.BytesIO(img))
        temp_image.save(temp_image_path)
        
        # 重新加载图片并设置 name 属性
        temp_image = Image.open(temp_image_path)
        temp_image.name = temp_image_path
        
        # 使用流式翻译，传递 PIL Image 对象
        # 注意：流式响应中不能在 finally 中删除文件，因为响应还在进行中
        # 临时文件会在 result 目录中累积，需要定期清理
        return await while_streaming(req, transform_to_image, conf, temp_image, "load_text")
    
    except Exception as e:
        # 只在出错时清理临时文件
        try:
            if os.path.exists(json_path):
                os.unlink(json_path)
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
        except:
            pass  # 忽略清理错误
        raise

@app.post("/translate/export/original/stream", response_class=StreamingResponse, tags=["api", "export", "stream"])
async def export_original_stream(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """导出原文（流式，支持进度）"""
    img = await image.read()
    conf = parse_config(config)
    return await while_streaming(req, transform_to_json, conf, img, "export_original")

@app.post("/translate/export/translated/stream", response_class=StreamingResponse, tags=["api", "export", "stream"])
async def export_translated_stream(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """导出译文（流式，支持进度）"""
    img = await image.read()
    conf = parse_config(config)
    return await while_streaming(req, transform_to_json, conf, img, "save_json")

@app.post("/translate/upscale/stream", response_class=StreamingResponse, tags=["api", "process", "stream"])
async def upscale_only_stream(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """仅超分（流式，支持进度）"""
    img = await image.read()
    conf = parse_config(config)
    return await while_streaming(req, transform_to_image, conf, img, "upscale_only")

@app.post("/translate/colorize/stream", response_class=StreamingResponse, tags=["api", "process", "stream"])
async def colorize_only_stream(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """仅上色（流式，支持进度）"""
    img = await image.read()
    conf = parse_config(config)
    return await while_streaming(req, transform_to_image, conf, img, "colorize_only")

@app.post("/translate/inpaint/stream", response_class=StreamingResponse, tags=["api", "process", "stream"])
async def inpaint_only_stream(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """仅修复（流式，支持进度）"""
    img = await image.read()
    conf = parse_config(config)
    return await while_streaming(req, transform_to_image, conf, img, "inpaint_only")



@app.post("/translate/complete", tags=["api", "form"])
async def translate_complete(req: Request, image: UploadFile = File(...), config: str = Form("{}")):
    """翻译图片，返回完整结果（JSON + 图片 + TXT）以 multipart 形式"""
    workflow = "normal"
    import json
    from fastapi.responses import Response
    
    img = await image.read()
    conf = parse_config(config)
    
    # 执行翻译
    ctx = await get_ctx(req, conf, img, workflow)
    
    # 获取 JSON 数据
    translation_data = to_translation(ctx)
    json_str = translation_data.model_dump_json()
    
    # 获取图片数据
    img_byte_arr = io.BytesIO()
    if ctx.result:
        ctx.result.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()
    
    # 构建 multipart 响应
    boundary = "----WebKitFormBoundary" + secrets.token_hex(16)
    
    parts = []
    
    # Part 1: JSON
    parts.append(f'--{boundary}\r\n')
    parts.append('Content-Disposition: form-data; name="json"\r\n')
    parts.append('Content-Type: application/json\r\n\r\n')
    parts.append(json_str)
    parts.append('\r\n')
    
    # Part 2: Image
    parts.append(f'--{boundary}\r\n')
    parts.append('Content-Disposition: form-data; name="image"; filename="result.png"\r\n')
    parts.append('Content-Type: image/png\r\n\r\n')
    
    # 组合响应
    response_parts = []
    for part in parts:
        if isinstance(part, str):
            response_parts.append(part.encode('utf-8'))
        else:
            response_parts.append(part)
    
    response_parts.append(img_bytes)
    response_parts.append(f'\r\n--{boundary}--\r\n'.encode('utf-8'))
    
    response_body = b''.join(response_parts)
    
    return Response(
        content=response_body,
        media_type=f'multipart/form-data; boundary={boundary}',
        headers={
            "Content-Length": str(len(response_body))
        }
    )

@app.get("/", tags=["info"])
async def root():
    """Web UI 主页"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        # 如果没有 Web UI，返回 API 信息
        return {
            "message": "Manga Translator API Server",
            "version": "2.0",
            "endpoints": {
                "translate": "/translate/image",
                "translate_stream": "/translate/with-form/image/stream",
                "batch": "/translate/batch/json",
                "docs": "/docs"
            }
        }

@app.get("/api", tags=["info"])
async def api_info():
    """API 服务器信息"""
    return {
        "message": "Manga Translator API Server",
        "version": "2.0",
        "endpoints": {
            "translate": "/translate/image",
            "translate_stream": "/translate/with-form/image/stream",
            "batch": "/translate/batch/json",
            "docs": "/docs"
        }
    }

def generate_nonce():
    return secrets.token_hex(16)

def start_translator_client_proc(host: str, port: int, nonce: str, params: Namespace):
    cmds = [
        sys.executable,
        '-m', 'manga_translator',
        'shared',
        '--host', host,
        '--port', str(port),
        '--nonce', nonce,
    ]
    if params.use_gpu:
        cmds.append('--use-gpu')
    if params.use_gpu_limited:
        cmds.append('--use-gpu-limited')
    if params.ignore_errors:
        cmds.append('--ignore-errors')
    if params.verbose:
        cmds.append('--verbose')
    if params.models_ttl:
        cmds.append('--models-ttl=%s' % params.models_ttl)
    if getattr(params, 'pre_dict', None):
        cmds.extend(['--pre-dict', params.pre_dict])
    if getattr(params, 'post_dict', None):
        cmds.extend(['--post-dict', params.post_dict])       
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(base_path)
    proc = subprocess.Popen(cmds, cwd=parent)
    executor_instances.register(ExecutorInstance(ip=host, port=port))

    def handle_exit_signals(signal, frame):
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit_signals)
    signal.signal(signal.SIGTERM, handle_exit_signals)

    return proc

def prepare(args):
    global nonce
    
    # web 模式没有 nonce 参数，使用 getattr 避免 AttributeError
    args_nonce = getattr(args, 'nonce', None)
    if args_nonce is None:
        nonce = os.getenv('MT_WEB_NONCE', generate_nonce())
    else:
        nonce = args_nonce
    
    # start_instance 也可能不存在于某些模式
    if getattr(args, 'start_instance', False):
        return start_translator_client_proc(args.host, args.port + 1, nonce, args)
    
    folder_name= "upload-cache"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

@app.post("/simple_execute/translate_batch", tags=["internal-api"])
async def simple_execute_batch(req: Request, data: BatchTranslateRequest):
    """Internal batch translation execution endpoint"""
    # Implementation for batch translation logic
    # Currently returns empty results, actual implementation needs to call batch translator
    from manga_translator import MangaTranslator
    translator = MangaTranslator({'batch_size': data.batch_size})
    
    # Prepare image-config pairs
    images_with_configs = [(img, data.config) for img in data.images]
    
    # Execute batch translation
    results = await translator.translate_batch(images_with_configs, data.batch_size)
    
    return results

@app.post("/execute/translate_batch", tags=["internal-api"])
async def execute_batch_stream(req: Request, data: BatchTranslateRequest):
    """Internal batch translation streaming execution endpoint"""
    # Streaming batch translation implementation
    from manga_translator import MangaTranslator
    translator = MangaTranslator({'batch_size': data.batch_size})
    
    # Prepare image-config pairs
    images_with_configs = [(img, data.config) for img in data.images]
    
    # Execute batch translation (streaming version requires more complex implementation)
    results = await translator.translate_batch(images_with_configs, data.batch_size)
    
    return results

@app.get("/results/list", tags=["api"])
async def list_results():
    """List all result directories"""
    result_dir = "../result"
    if not os.path.exists(result_dir):
        return {"directories": []}
    
    try:
        directories = []
        for item in os.listdir(result_dir):
            item_path = os.path.join(result_dir, item)
            if os.path.isdir(item_path):
                # Check if final.png exists in this directory
                final_png_path = os.path.join(item_path, "final.png")
                if os.path.exists(final_png_path):
                    directories.append(item)
        return {"directories": directories}
    except Exception as e:
        raise HTTPException(500, detail=f"Error listing results: {str(e)}")

@app.delete("/results/clear", tags=["api"])
async def clear_results():
    """Delete all result directories"""
    result_dir = "../result"
    if not os.path.exists(result_dir):
        return {"message": "No results directory found"}
    
    try:
        deleted_count = 0
        for item in os.listdir(result_dir):
            item_path = os.path.join(result_dir, item)
            if os.path.isdir(item_path):
                # Check if final.png exists in this directory
                final_png_path = os.path.join(item_path, "final.png")
                if os.path.exists(final_png_path):
                    shutil.rmtree(item_path)
                    deleted_count += 1
        
        return {"message": f"Deleted {deleted_count} result directories"}
    except Exception as e:
        raise HTTPException(500, detail=f"Error clearing results: {str(e)}")

@app.delete("/results/{folder_name}", tags=["api"])
async def delete_result(folder_name: str):
    """Delete a specific result directory"""
    result_dir = "../result"
    folder_path = os.path.join(result_dir, folder_name)
    
    if not os.path.exists(folder_path):
        raise HTTPException(404, detail="Result directory not found")
    
    try:
        # Check if final.png exists in this directory
        final_png_path = os.path.join(folder_path, "final.png")
        if not os.path.exists(final_png_path):
            raise HTTPException(404, detail="Result file not found")
        
        shutil.rmtree(folder_path)
        return {"message": f"Deleted result directory: {folder_name}"}
    except Exception as e:
        raise HTTPException(500, detail=f"Error deleting result: {str(e)}")

#todo: restart if crash
#todo: cache results
#todo: cleanup cache

@app.post("/cleanup/temp", tags=["api", "maintenance"])
async def cleanup_temp_files(max_age_hours: int = 24):
    """
    清理临时文件
    
    Args:
        max_age_hours: 清理多少小时前的临时文件（默认24小时）
    
    Returns:
        清理结果统计
    """
    import time
    
    result_dir = "result"
    if not os.path.exists(result_dir):
        return {"deleted": 0, "message": "No temp directory found"}
    
    deleted_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    try:
        for filename in os.listdir(result_dir):
            if filename.startswith("temp_"):
                filepath = os.path.join(result_dir, filename)
                try:
                    # 检查文件年龄
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > max_age_seconds:
                        if os.path.isfile(filepath):
                            os.unlink(filepath)
                            deleted_count += 1
                        elif os.path.isdir(filepath):
                            shutil.rmtree(filepath)
                            deleted_count += 1
                except Exception as e:
                    # 忽略单个文件的删除错误（可能被占用）
                    continue
        
        return {
            "deleted": deleted_count,
            "message": f"Successfully cleaned up {deleted_count} temporary files older than {max_age_hours} hours"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error during cleanup: {str(e)}")

def init_translator(use_gpu=False, verbose=False):
    """初始化翻译器（预留函数）"""
    # 这个函数用于预加载模型等初始化操作
    # 目前翻译器在首次请求时才会初始化
    pass

def run_server(args):
    """启动 Web API 服务器（纯API模式，不带界面）"""
    import uvicorn
    
    global nonce, server_config
    
    # 先设置服务器配置（在 prepare 之前）
    server_config['use_gpu'] = getattr(args, 'use_gpu', False)
    server_config['use_gpu_limited'] = getattr(args, 'use_gpu_limited', False)
    server_config['verbose'] = getattr(args, 'verbose', False)
    server_config['models_ttl'] = getattr(args, 'models_ttl', 0)
    server_config['retry_attempts'] = getattr(args, 'retry_attempts', None)
    
    # 从 admin_settings 加载管理员密码和并发设置
    server_config['admin_password'] = admin_settings.get('admin_password')
    if admin_settings.get('max_concurrent_tasks'):
        server_config['max_concurrent_tasks'] = admin_settings['max_concurrent_tasks']
    
    print(f"[SERVER CONFIG] use_gpu={server_config['use_gpu']}, use_gpu_limited={server_config['use_gpu_limited']}, verbose={server_config['verbose']}, models_ttl={server_config['models_ttl']}, retry_attempts={server_config['retry_attempts']}, max_concurrent_tasks={server_config['max_concurrent_tasks']}")
    
    # 初始化并发控制
    init_semaphore()
    
    # web 模式不启动独立的翻译实例（与旧版本保持一致）
    args.start_instance = False
    proc = prepare(args)
    print("Nonce: "+nonce)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception:
        if proc:
            proc.terminate()

def main(args):
    """启动 Web UI 服务器（带界面模式）"""
    # ui 模式和 web 模式使用相同的实现
    run_server(args)

if __name__ == '__main__':
    from args import parse_arguments
    args = parse_arguments()
    main(args)
