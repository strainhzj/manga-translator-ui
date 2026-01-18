import os
import asyncio
import json
from typing import List, Dict, Any
import aiohttp

from .common import CommonTranslator, VALID_LANGUAGES, parse_json_or_text_response, parse_hq_response, get_glossary_extraction_prompt, merge_glossary_to_file, sanitize_text_encoding
from .keys import VERTEX_API_KEY, VERTEX_MODEL
from ..utils import Context


# Vertex AI API 配置
VERTEX_API_BASE = "https://aiplatform.googleapis.com/v1/publishers/google/models"


class VertexTranslator(CommonTranslator):
    """
    Google Vertex AI 翻译器（使用 Gemini 模型）
    支持批量文本翻译，使用 REST API（不包含图片处理）
    """
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES

    # 类变量: 跨实例共享的RPM限制时间戳
    _GLOBAL_LAST_REQUEST_TS = {}  # {model_name: timestamp}

    def __init__(self):
        super().__init__()
        self.prev_context = ""  # 用于存储多页上下文

        # 只在非Web环境下重新加载.env文件
        is_web_server = os.getenv('MANGA_TRANSLATOR_WEB_SERVER', 'false').lower() == 'true'
        if not is_web_server:
            from dotenv import load_dotenv
            load_dotenv(override=True)

        self.api_key = os.getenv('VERTEX_API_KEY', VERTEX_API_KEY)
        self.model_name = os.getenv('VERTEX_MODEL', VERTEX_MODEL)
        self.max_tokens = None  # 不限制，使用模型默认最大值
        self.temperature = 0.1
        self._MAX_REQUESTS_PER_MINUTE = 0  # 默认无限制

        # 验证 API Key 格式
        if self.api_key:
            self._validate_api_key_format()

        # 读取代理配置
        self.proxy = None
        https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
        http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')

        # Vertex AI 使用 HTTPS，优先使用 HTTPS_PROXY，回退到 HTTP_PROXY
        if https_proxy:
            self.proxy = https_proxy
            self.logger.info(f"使用代理: {https_proxy}")
        elif http_proxy:
            self.proxy = http_proxy
            self.logger.info(f"使用代理: {http_proxy}")

        # 使用全局时间戳,跨实例共享
        if self.model_name not in VertexTranslator._GLOBAL_LAST_REQUEST_TS:
            VertexTranslator._GLOBAL_LAST_REQUEST_TS[self.model_name] = 0
        self._last_request_ts_key = self.model_name

    def set_prev_context(self, context: str):
        """设置多页上下文（用于context_size > 0时）"""
        self.prev_context = context if context else ""

    def _validate_api_key_format(self):
        """验证 API Key 格式，提供友好的错误提示"""
        if not self.api_key:
            return

        # 基本检查：API Key 不应为空
        if len(self.api_key) < 10:
            self.logger.warning("=== Vertex AI API Key 格式检查 ===")
            self.logger.warning("API Key 长度异常，请检查是否完整复制")
            self.logger.warning("====================================")
            self.logger.warning("")

        # 注意：不再对 API Key 格式做过度限制
        # Google Cloud API Key 可能有多种格式，只要用户确认是正确的即可

    def parse_args(self, args):
        """解析配置参数"""
        # 调用父类的 parse_args 来设置通用参数（包括 attempts、post_check 等）
        super().parse_args(args)

        # 同步 attempts 到 _max_total_attempts
        self._max_total_attempts = self.attempts

        # 从配置中读取RPM限制
        max_rpm = getattr(args, 'max_requests_per_minute', 0)
        if max_rpm > 0:
            self._MAX_REQUESTS_PER_MINUTE = max_rpm
            self.logger.info(f"Setting Vertex AI max requests per minute to: {max_rpm}")

        # 从配置中读取用户级 API Key（优先于环境变量）
        user_api_key = getattr(args, 'user_api_key', None)
        if user_api_key and user_api_key != self.api_key:
            self.api_key = user_api_key
            self.logger.info("[UserAPIKey] Using user-provided API key for Vertex AI")

        user_api_model = getattr(args, 'user_api_model', None)
        if user_api_model:
            self.model_name = user_api_model
            # 更新全局时间戳的 key
            if self.model_name not in VertexTranslator._GLOBAL_LAST_REQUEST_TS:
                VertexTranslator._GLOBAL_LAST_REQUEST_TS[self.model_name] = 0
            self._last_request_ts_key = self.model_name
            self.logger.info(f"[UserAPIKey] Using user-provided model: {user_api_model}")

    def _build_api_url(self) -> str:
        """构建 API URL"""
        # Vertex AI 使用 streamGenerateContent endpoint
        url = f"{VERTEX_API_BASE}/{self.model_name}:streamGenerateContent"
        # 添加 API Key 作为查询参数
        if self.api_key:
            url += f"?key={self.api_key}"
        return url

    def _flatten_prompt_data(self, data: Any, indent: int = 0) -> str:
        """递归地将字典或列表扁平化为格式化字符串"""
        prompt_parts = []
        prefix = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    prompt_parts.append(f"{prefix}- {key}:")
                    prompt_parts.append(self._flatten_prompt_data(value, indent + 1))
                else:
                    prompt_parts.append(f"{prefix}- {key}: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    prompt_parts.append(self._flatten_prompt_data(item, indent + 1))
                else:
                    prompt_parts.append(f"{prefix}- {item}")

        return "\n".join(prompt_parts)

    def _build_system_prompt(self, source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, retry_attempt: int = 0, retry_reason: str = "", extract_glossary: bool = False) -> str:
        """构建系统提示词"""
        # Map language codes to full names for clarity in the prompt
        lang_map = {
            "CHS": "Simplified Chinese",
            "CHT": "Traditional Chinese",
            "JPN": "Japanese",
            "ENG": "English",
            "KOR": "Korean",
            "VIN": "Vietnamese",
            "FRA": "French",
            "DEU": "German",
            "ITA": "Italian",
        }
        target_lang_full = lang_map.get(target_lang, target_lang)

        custom_prompt_str = ""
        if custom_prompt_json:
            custom_prompt_str = self._flatten_prompt_data(custom_prompt_json)

        line_break_prompt_str = ""
        if line_break_prompt_json and line_break_prompt_json.get('line_break_prompt'):
            line_break_prompt_str = line_break_prompt_json['line_break_prompt']

        # 尝试加载 HQ System Prompt
        base_prompt = ""
        try:
            from ..utils import BASE_PATH
            import os
            import json
            prompt_path = os.path.join(BASE_PATH, 'dict', 'system_prompt_hq.json')
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    base_prompt_data = json.load(f)
                base_prompt = base_prompt_data['system_prompt']
        except Exception as e:
            self.logger.warning(f"Failed to load system prompt from file: {e}")

        # 如果没加载到 HQ Prompt，使用简单的 fallback
        if not base_prompt:
             base_prompt = f"""You are an expert manga translator. Translate from {source_lang} to {target_lang}. Output only the translation."""

        # Replace placeholder with the full language name
        base_prompt = base_prompt.replace("{{{target_lang}}}", target_lang_full)

        # Also replace target_lang placeholder in custom prompt
        if custom_prompt_str:
            custom_prompt_str = custom_prompt_str.replace("{{{target_lang}}}", target_lang_full)

        # Combine prompts
        final_prompt = ""

        # 添加重试提示到最前面（如果是重试）
        if retry_attempt > 0:
            final_prompt += self._get_retry_hint(retry_attempt, retry_reason) + "\n"

        if line_break_prompt_str:
            final_prompt += f"{line_break_prompt_str}\n\n---\n\n"
        if custom_prompt_str:
            final_prompt += f"{custom_prompt_str}\n\n---\n\n"

        final_prompt += base_prompt

        # 追加术语提取提示词
        if extract_glossary:
            extraction_prompt = get_glossary_extraction_prompt(target_lang_full)
            if extraction_prompt:
                final_prompt += f"\n\n---\n\n{extraction_prompt}"
                self.logger.info("已启用自动术语提取，提示词已追加。")

        return final_prompt

    def _build_user_prompt(self, texts: List[str], ctx: Any, retry_attempt: int = 0, retry_reason: str = "") -> str:
        """构建用户提示词（纯文本版）- 使用 JSON 格式以配合 HQ Prompt"""
        return self._build_user_prompt_for_texts(texts, ctx, self.prev_context, retry_attempt=retry_attempt, retry_reason=retry_reason)

    def _get_system_instruction(self, source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, retry_attempt: int = 0, retry_reason: str = "", extract_glossary: bool = False) -> str:
        """获取完整的系统指令"""
        return self._build_system_prompt(source_lang, target_lang, custom_prompt_json=custom_prompt_json, line_break_prompt_json=line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)

    async def _translate_batch(self, texts: List[str], source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, ctx: Any = None, split_level: int = 0) -> List[str]:
        """批量翻译方法（纯文本）"""
        if not texts:
            return []

        if not self.api_key:
            self.logger.error("Vertex AI API key not configured")
            raise Exception("Vertex AI API key not configured. Please set VERTEX_API_KEY environment variable.")

        # 初始化重试信息
        retry_attempt = 0
        retry_reason = ""

        # 保存参数供重试时使用
        _source_lang = source_lang
        _target_lang = target_lang
        _custom_prompt_json = custom_prompt_json
        _line_break_prompt_json = line_break_prompt_json

        # 发送请求
        max_retries = self.attempts
        attempt = 0
        is_infinite = max_retries == -1
        last_exception = None
        local_attempt = 0  # 本次批次的尝试次数

        while is_infinite or attempt < max_retries:
            # 检查是否被取消
            self._check_cancelled()

            # 检查全局尝试次数
            if not self._increment_global_attempt():
                self.logger.error("Reached global attempt limit. Stopping translation.")
                last_error_msg = str(last_exception) if last_exception else "Unknown error"
                raise Exception(f"达到最大尝试次数 ({self._max_total_attempts})，最后一次错误: {last_error_msg}")

            local_attempt += 1
            attempt += 1

            # 确定是否开启术语提取
            config_extract = False
            if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                config_extract = getattr(ctx.config.translator, 'extract_glossary', False)

            extract_glossary = bool(_custom_prompt_json) and config_extract

            # 获取系统指令
            system_instruction = self._get_system_instruction(_source_lang, _target_lang, custom_prompt_json=_custom_prompt_json, line_break_prompt_json=_line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)

            # 构建用户提示词
            user_prompt = self._build_user_prompt(texts, ctx, retry_attempt=retry_attempt, retry_reason=retry_reason)

            # 将系统提示词合并到用户消息的开头
            combined_prompt = f"{system_instruction}\n\n{user_prompt}"

            # 动态调整温度
            current_temperature = self._get_retry_temperature(self.temperature, retry_attempt, retry_reason)

            # 构建请求体（严格遵循 Vertex AI curl 示例格式）
            # 基础请求体（匹配官方示例）
            request_data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": combined_prompt}
                        ]
                    }
                ]
            }

            # generationConfig 是可选的，只在需要时添加
            # 这样可以完全匹配 curl 示例的格式
            if current_temperature != 0.1 or self.max_tokens:
                request_data["generationConfig"] = {}
                if current_temperature != 0.1:
                    request_data["generationConfig"]["temperature"] = current_temperature
                if self.max_tokens:
                    request_data["generationConfig"]["maxOutputTokens"] = self.max_tokens

            try:
                # RPM限制
                if self._MAX_REQUESTS_PER_MINUTE > 0:
                    import time
                    now = time.time()
                    delay = 60.0 / self._MAX_REQUESTS_PER_MINUTE
                    elapsed = now - VertexTranslator._GLOBAL_LAST_REQUEST_TS[self._last_request_ts_key]
                    if elapsed < delay:
                        sleep_time = delay - elapsed
                        self.logger.info(f'Ratelimit sleep: {sleep_time:.2f}s')
                        await asyncio.sleep(sleep_time)

                if retry_attempt > 0 and current_temperature != self.temperature:
                    self.logger.info(f"[重试] 温度调整: {self.temperature} -> {current_temperature}")

                # 发送 HTTP 请求
                url = self._build_api_url()
                headers = {
                    "Content-Type": "application/json",
                }

                # 始终打印请求详情（用于调试）
                self.logger.info(f"=== Vertex AI 请求详情 ===")
                self.logger.info(f"URL: {url}")
                self.logger.info(f"Method: POST")
                self.logger.info(f"Headers: {headers}")
                try:
                    body_str = json.dumps(request_data, ensure_ascii=False, indent=2)
                    self.logger.info(f"Body:\n{body_str}")
                except Exception as e:
                    self.logger.error(f"Body 序列化失败: {e}")
                    self.logger.info(f"Body (raw): {request_data}")
                self.logger.info(f"========================")

                # 创建会话，如果配置了代理则使用代理
                # 注意：禁用 auto_decompress，手动处理 gzip 解压
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=request_data, headers=headers, proxy=self.proxy,
                                          auto_decompress=False) as response:
                        if response.status != 200:
                            error_text = await response.text()

                            # 针对 401 错误提供详细的诊断信息
                            if response.status == 401:
                                self.logger.error("=== Vertex AI 认证失败 (401) ===")
                                self.logger.error("可能的原因：")
                                self.logger.error("  1. API Key 无效或过期")
                                self.logger.error("  2. 使用了错误的 API Key 类型（如 OAuth token）")
                                self.logger.error("  3. Google Cloud 项目未启用 Vertex AI API")
                                self.logger.error("")
                                self.logger.error("诊断步骤：")
                                self.logger.error("  1. 检查 API Key 是否正确复制（无多余空格）")
                                self.logger.error("  2. 确认 API Key 来自 Google Cloud Console，不是 Google AI Studio")
                                self.logger.error("  3. 验证 Google Cloud 项目已启用 Vertex AI API")
                                self.logger.error("  4. 检查网络连接（可能需要配置代理）")
                                self.logger.error("")
                                self.logger.error("详细配置指南：doc/VERTEX_AI_CONFIG.md")
                                self.logger.error("================================")

                            raise Exception(f"Vertex AI API error: {response.status} - {error_text}")

                        # === 手动处理 gzip 压缩和 SSE 格式 ===
                        # 一次性读取所有数据（已禁用自动解压）
                        raw_data = await response.read()
                        self.logger.info(f"=== Vertex AI 响应处理 ===")
                        self.logger.info(f"原始数据长度: {len(raw_data)} 字节")
                        self.logger.info(f"前 16 字节 (hex): {raw_data[:16].hex()}")

                        # 检测并处理 gzip 压缩
                        if len(raw_data) >= 2 and raw_data[:2] == b'\x1f\x8b':
                            import gzip
                            self.logger.info("检测到 gzip 压缩，正在解压...")
                            try:
                                decompressed = gzip.decompress(raw_data)
                                self.logger.info(f"gzip 解压成功: {len(raw_data)} -> {len(decompressed)} 字节")
                                raw_data = decompressed
                            except Exception as e:
                                self.logger.error(f"gzip 解压失败: {e}")
                                raise Exception(f"Failed to decompress gzip response: {e}")
                        else:
                            self.logger.info("未检测到 gzip 压缩，直接处理")

                        # 解码为文本
                        text_response = raw_data.decode('utf-8', errors='replace')
                        self.logger.info(f"解码后文本长度: {len(text_response)} 字符")
                        self.logger.info(f"前 500 字符: {text_response[:500]}")
                        self.logger.info(f"==========================")

                        # Vertex AI 返回的是 JSON 数组格式，不是 SSE 格式
                        result_text = ""

                        # 尝试直接解析为 JSON 数组
                        try:
                            self.logger.info("尝试解析为 JSON 数组格式...")
                            chunks = json.loads(text_response)
                            self.logger.info(f"成功解析 JSON 数组，包含 {len(chunks)} 个元素")

                            for idx, chunk_data in enumerate(chunks):
                                if 'candidates' in chunk_data and chunk_data['candidates']:
                                    candidate = chunk_data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                result_text += part['text']
                                                self.logger.debug(f"提取文本 [块 {idx}]: {part['text'][:50]}...")
                        except json.JSONDecodeError:
                            # 如果失败，尝试 SSE 格式解析
                            self.logger.info("JSON 数组解析失败，尝试 SSE 格式...")
                            lines = text_response.split('\n')
                            self.logger.info(f"分割成 {len(lines)} 行")

                            for line_str in lines:
                                line_str = line_str.strip()
                                if not line_str:
                                    continue

                                # SSE 格式：每行以 "data:" 开头
                                if line_str.startswith('data:'):
                                    json_str = line_str[5:].strip()
                                    try:
                                        chunk_data = json.loads(json_str)
                                        if 'candidates' in chunk_data and chunk_data['candidates']:
                                            candidate = chunk_data['candidates'][0]
                                            if 'content' in candidate and 'parts' in candidate['content']:
                                                for part in candidate['content']['parts']:
                                                    if 'text' in part:
                                                        result_text += part['text']
                                                        self.logger.debug(f"提取文本 (SSE): {part['text'][:50]}...")
                                    except json.JSONDecodeError as e:
                                        self.logger.debug(f"跳过非 JSON 行: {json_str[:100]} (error: {e})")
                                        continue

                        # 诊断日志
                        self.logger.info(f"=== Vertex AI 响应统计 ===")
                        self.logger.info(f"最终文本长度: {len(result_text)}")
                        if result_text:
                            self.logger.info(f"文本预览: {result_text[:200]}")
                        self.logger.info(f"========================")

                if self._MAX_REQUESTS_PER_MINUTE > 0:
                    import time
                    VertexTranslator._GLOBAL_LAST_REQUEST_TS[self._last_request_ts_key] = time.time()

                if not result_text:
                    self.logger.error("=== Vertex AI 空响应诊断 ===")
                    self.logger.error("API 返回了响应，但未能提取到文本内容")
                    self.logger.error("可能的原因：")
                    self.logger.error("  1. 响应格式与预期不符（非 SSE 格式）")
                    self.logger.error("  2. JSON 结构中缺少 'content/parts/text' 字段")
                    self.logger.error("  3. 数据编码问题")
                    self.logger.error("")
                    self.logger.error("请检查上方的详细日志了解响应内容")
                    self.logger.error("========================")
                    raise Exception("Empty response from Vertex AI API")

                result_text = result_text.strip()

                # 统一的编码清理（处理UTF-16-LE等编码问题）
                result_text = sanitize_text_encoding(result_text)

                self.logger.debug(f"--- Vertex AI Raw Response ---\n{result_text}\n---------------------------")

                # 使用 parse_hq_response 解析（支持 JSON Object/Array/Text）
                translations, new_terms = parse_hq_response(result_text)

                # 处理术语提取
                if extract_glossary and new_terms:
                    prompt_path = None
                    if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                        prompt_path = getattr(ctx.config.translator, 'high_quality_prompt_path', None)

                    if prompt_path:
                        merge_glossary_to_file(prompt_path, new_terms)
                    else:
                        self.logger.warning("Extracted new terms but prompt path not found in context.")

                # Strict validation: must match input count
                if len(translations) != len(texts):
                    retry_attempt += 1
                    retry_reason = f"Translation count mismatch: expected {len(texts)}, got {len(translations)}"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}. Retrying...")
                    self.logger.warning(f"Expected texts: {texts}")
                    self.logger.warning(f"Got translations: {translations}")

                    # 记录错误以便在达到最大尝试次数时显示
                    last_exception = Exception(f"翻译数量不匹配: 期望 {len(texts)} 条，实际得到 {len(translations)} 条")

                    if not is_infinite and attempt >= max_retries:
                        raise Exception(f"Translation count mismatch after {max_retries} attempts: expected {len(texts)}, got {len(translations)}")

                    await asyncio.sleep(2)
                    continue

                # 质量验证：检查空翻译、合并翻译、可疑符号等
                is_valid, error_msg = self._validate_translation_quality(texts, translations)
                if not is_valid:
                    retry_attempt += 1
                    retry_reason = f"Quality check failed: {error_msg}"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}. Retrying...")

                    # 记录错误以便在达到最大尝试次数时显示
                    last_exception = Exception(f"翻译质量检查失败: {error_msg}")

                    if not is_infinite and attempt >= max_retries:
                        raise Exception(f"Quality check failed after {max_retries} attempts: {error_msg}")

                    await asyncio.sleep(2)
                    continue

                # 打印原文和译文的对应关系
                self.logger.info("--- Translation Results ---")
                for original, translated in zip(texts, translations):
                    self.logger.info(f'{original} -> {translated}')
                self.logger.info("---------------------------")

                # BR检查：检查翻译结果是否包含必要的[BR]标记
                if not self._validate_br_markers(translations, queries=texts, ctx=ctx):
                    retry_attempt += 1
                    retry_reason = "BR markers missing in translations"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}, retrying...")

                    # 记录错误以便在达到最大尝试次数时显示
                    last_exception = Exception("AI断句检查失败: 翻译结果缺少必要的[BR]标记")

                    # 如果达到最大重试次数，抛出友好的异常
                    if not is_infinite and attempt >= max_retries:
                        from .common import BRMarkersValidationException
                        self.logger.error("Vertex AI翻译在多次重试后仍然失败。")
                        raise BRMarkersValidationException(
                            missing_count=0,
                            total_count=len(texts),
                            tolerance=max(1, len(texts) // 10)
                        )
                    await asyncio.sleep(2)
                    continue

                return translations[:len(texts)]

            except Exception as e:
                error_message = str(e)
                last_exception = e  # 保存最后一次错误

                attempt += 1
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                self.logger.warning(f"Vertex AI翻译出错 ({log_attempt}): {e}")

                # 检查是否达到最大重试次数（注意：attempt已经+1了）
                if not is_infinite and attempt >= max_retries:
                    self.logger.error("Vertex AI翻译在多次重试后仍然失败。即将终止程序。")
                    raise e

                await asyncio.sleep(1)

        return texts

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        """主翻译方法"""
        if not queries:
            return []

        # 重置全局尝试计数器
        self._reset_global_attempt_count()

        self.logger.info(f"使用 Vertex AI 纯文本翻译模式处理{len(queries)}个文本，最大尝试次数: {self._max_total_attempts}")
        custom_prompt_json = getattr(ctx, 'custom_prompt_json', None) if ctx else None
        line_break_prompt_json = getattr(ctx, 'line_break_prompt_json', None) if ctx else None

        # 使用分割包装器进行翻译
        translations = await self._translate_with_split(
            self._translate_batch,
            queries,
            split_level=0,
            source_lang=from_lang,
            target_lang=to_lang,
            custom_prompt_json=custom_prompt_json,
            line_break_prompt_json=line_break_prompt_json,
            ctx=ctx
        )

        # 应用文本后处理
        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]
        return translations
