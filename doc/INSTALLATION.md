# 安装指南

本文档提供详细的安装步骤和系统要求说明。

---

## 📋 目录

- [系统要求](#系统要求)
- [安装方式一：使用安装脚本（推荐，支持自动更新）](#安装方式一使用安装脚本推荐支持自动更新)
- [安装方式二：下载打包版本](#安装方式二下载打包版本)
- [安装方式三：从源码运行](#安装方式三从源码运行)
- [安装方式四：Docker部署](#安装方式四docker部署)
- [故障排除](#故障排除)

---

## 系统要求

### 最低配置

- **操作系统**：Windows 10/11 (64位) 或 Linux
- **内存**：8 GB RAM
- **存储空间**：5 GB 可用空间（用于程序和模型文件）
- **Python 版本**（开发版）：Python 3.12

### 推荐配置

- **内存**：16 GB RAM 或更多
- **GPU**：
  - **NVIDIA 显卡**：支持 CUDA 12.x（需驱动版本 >= 525.60.13）
    - 建议显存：6 GB 或更多
    - 支持的 NVIDIA 显卡：GTX 1060 及以上
  - **AMD 显卡**：支持 ROCm（实验性）
    - 支持的显卡：**仅 RX 7000/9000 系列（RDNA 3/4）**
    - ⚠️ RX 5000/6000 系列请使用 CPU 版本
    - ⚠️ AMD GPU 仅支持安装脚本方式，不支持打包版本
    - ⚠️ Windows 上 ROCm 支持有限，Linux 下体验更好
- **存储空间**：10 GB SSD

---

## 安装方式一：使用安装脚本（⭐ 推荐，自动安装 Miniconda）

脚本会自动完成所有配置，并支持一键更新。

> ⚠️ **网络提示**：下载过程需要从 GitHub 拉取代码，网络不好建议开代理。
> 💡 **新特性**：无需预装 Python，脚本会自动安装 Miniconda（轻量级Python环境管理）

### 前提条件

- **无需预装 Python**：脚本会自动下载安装 Miniconda
- **Git**（可选）：脚本可以自动下载便携版 Git

### 详细步骤

#### 1. 获取安装脚本

- 访问仓库：[https://github.com/hgmzhn/manga-translator-ui](https://github.com/hgmzhn/manga-translator-ui)
- 下载 [`步骤1-首次安装.bat`](https://github.com/hgmzhn/manga-translator-ui/raw/main/步骤1-首次安装.bat)
- 保存到你想安装程序的目录（如 `D:\manga-translator-ui\`）

#### 2. 运行安装脚本

双击 `步骤1-首次安装.bat`，脚本会：

**2.1 检测并安装 Miniconda**
- ✓ 如果系统已有 Python/Conda，直接使用
- ✗ 如果未安装：
  - 提供下载源选择：清华大学镜像（国内推荐）或 Anaconda 官方
  - 自动下载 Miniconda3 安装程序（约 50MB）
  - 静默安装到：`<项目目录>\Miniconda3`（不占用C盘）
  - 自动配置环境变量
  - **注意**：安装完成后需要重新运行脚本（重新加载环境变量）

**2.2 检测/安装 Git**
- ✓ 如果系统已有 Git，使用系统 Git
- ✗ 如果没有 Git，提供两个选项：
  - **选项 1**（推荐）：自动下载便携版 Git（约 50MB）
  - **选项 2**：手动安装 Git 后重新运行

**2.3 选择下载源**
- **选项 1**：GitHub 官方源（国外网络）
- **选项 2**（推荐）：gh-proxy.com 镜像（国内更快）

**2.4 克隆/更新代码**
- 如果是首次安装：从 GitHub 克隆代码
- 如果已有代码：自动更新到最新版本

**2.5 创建 Conda 环境**
- 在项目目录创建 `conda_env` 环境（Python 3.12）
- 位置：`<项目目录>\conda_env\`
- **不占用C盘系统空间**，环境在项目目录内
- 隔离项目依赖，不影响系统

**2.6 安装依赖**
- 自动检测 GPU：
  - ✓ **NVIDIA 显卡**：
    - 检测 CUDA 版本
    - CUDA >= 12: 安装 GPU 版本依赖（requirements_gpu.txt）
    - CUDA < 12: 提示更新驱动或使用 CPU 版本
  - ✓ **AMD 显卡**：
    - 自动识别显卡型号和 gfx 版本
    - 询问用户确认后安装 AMD ROCm PyTorch（requirements_amd.txt）
    - **仅支持 RX 7000/9000 系列（RDNA 3/4）**
    - RX 5000/6000 系列会自动使用 CPU 版本
  - ✗ **其他显卡/集显**：安装 CPU 版本依赖（requirements_cpu.txt）
- 使用 `launch.py` 智能安装所有必需的包

**2.7 完成安装**
- 显示安装位置
- 询问是否立即运行程序

### Miniconda 特点

**优势：**
- ✅ 体积小（约 50MB）
- ✅ 可管理多个 Python 版本
- ✅ 环境隔离，互不干扰
- ✅ 自带 pip 包管理
- ✅ **完全安装在项目目录，不占用 C 盘系统空间**

**目录结构：**
```
D:\manga-translator-ui\          # 你选择的安装目录
├── 步骤1-首次安装.bat            # 安装脚本
├── 步骤2-启动Qt界面.bat          # 启动脚本
├── 步骤3-检查更新并启动.bat      # 更新并启动
├── 步骤4-更新维护.bat            # 维护工具
├── Miniconda3\                   # Miniconda主程序（约600MB）
│   ├── python.exe
│   ├── Scripts\
│   ├── pkgs\
│   └── ...
├── conda_env\                    # 项目虚拟环境（约2-5GB）
│   ├── python.exe
│   ├── Scripts\
│   ├── Lib\
│   └── ...
├── PortableGit\                  # 便携版Git（如果下载）
├── desktop_qt_ui\                # Qt界面源码
├── manga_translator\             # 核心翻译模块
└── ...                           # 其他项目文件
```

#### 3. 启动程序

安装完成后，以后每次使用只需：

双击 `步骤2-启动Qt界面.bat`

> **提示**：也可以双击 `步骤3-检查更新并启动.bat` 在启动前自动检查更新

#### 4. 更新程序（可选）

需要更新到最新版本时：

双击 `步骤4-更新维护.bat`，选择"完整更新"

---

## 安装方式二：下载打包版本

适合不想安装 Python 的用户，但文件较大（约 3-5 GB）。

### 1. 访问发布页面

前往 [GitHub Releases](https://github.com/hgmzhn/manga-translator-ui/releases) 页面。

### 2. 选择版本

下载最新版本的安装包：

**CPU 版本**：
- 文件名：`manga-translator-cpu-vX.X.X.zip` 或分卷文件
- 适用范围：所有电脑
- 优点：无需 GPU，兼容性好
- 缺点：翻译速度较慢

**GPU 版本**：
- 文件名：`manga-translator-gpu-vX.X.X.zip` 或分卷文件
- 适用范围：拥有 NVIDIA 显卡的电脑
- 要求：CUDA 12.x 支持
- 优点：翻译速度快
- 缺点：需要兼容的 NVIDIA 显卡

### 3. 分卷下载说明

如果文件被分成多个压缩包（如 `part1.rar`, `part2.rar`, `part3.rar`...），请按照以下步骤操作：

1. **下载所有分卷**：
   - 必须下载所有分卷文件到同一文件夹
   - 例如：`part1.rar`, `part2.rar`, `part3.rar`

2. **解压第一个分卷**：
   - 只需右键点击 `part1.rar`
   - 选择"解压到..."或"Extract to..."
   - 其他分卷会自动参与解压

3. **注意事项**：
   - 所有分卷必须在同一目录
   - 不要重命名分卷文件
   - 缺少任何一个分卷都会导致解压失败

### 4. 安装步骤

1. **解压文件**：
   ```
   将下载的压缩包解压到任意目录
   例如：D:\manga-translator\
   ```

2. **检查文件结构**：
   ```
   manga-translator/
   ├── app.exe          # 主程序
   ├── _internal/       # 依赖文件
   ├── fonts/           # 字体文件
   ├── models/          # AI 模型文件
   └── examples/        # 配置示例
   ```

3. **运行程序**：
   - 双击 `app.exe` 启动程序
   - 首次运行会自动加载模型文件

---

## 安装方式三：从源码运行

适合开发者或想自定义的用户。

### 1. 克隆仓库

```bash
git clone https://github.com/hgmzhn/manga-translator-ui.git
cd manga-translator-ui
```

### 2. 安装依赖

```bash
# CPU 版本
pip install -r requirements_cpu.txt

# GPU 版本（需要 CUDA 12.x）
pip install -r requirements_gpu.txt
```

### 3. 运行程序

```bash
# 运行 PyQt6 界面
python -m desktop_qt_ui.main

# 或运行旧版 CustomTkinter 界面
python -m desktop-ui.main
```

---

## 安装方式四：Docker部署

适合需要容器化部署或在服务器上运行的用户。

### 前提条件

- 已安装 Docker
- （可选）已安装 Docker Compose
- （GPU版本）已安装 NVIDIA Container Toolkit

### 环境变量配置

Docker部署支持通过环境变量配置服务器参数：

| 环境变量 | 说明 | 默认值 | 示例 |
|---------|------|--------|------|
| `MT_WEB_HOST` | 服务器监听地址 | `127.0.0.1` | `0.0.0.0` |
| `MT_WEB_PORT` | 服务器端口 | `8000` | `8080` |
| `MT_USE_GPU` | 是否使用GPU | `false` | `true` |
| `MT_MODELS_TTL` | 模型存活时间（秒） | `0`（永久） | `300` |
| `MT_RETRY_ATTEMPTS` | 翻译重试次数 | `-1`（无限） | `3` |
| `MT_VERBOSE` | 详细日志 | `false` | `true` |
| `MANGA_TRANSLATOR_ADMIN_PASSWORD` | 管理员密码（首次启动自动设置） | 无 | `your_password` |

**管理员密码说明**：
- 首次启动时，如果设置了 `MANGA_TRANSLATOR_ADMIN_PASSWORD` 环境变量，会自动设置为管理员密码
- 密码至少需要 6 位字符
- 密码会保存到 `admin_config.json`，后续启动不再读取环境变量
- 如需修改密码，请在管理面板中使用"更改管理员密码"功能

### 方式1：使用Docker命令

#### CPU版本

```bash
# 拉取镜像
docker pull your-registry/manga-translator:latest

# 运行容器（带管理员密码）
docker run -d \
  --name manga-translator \
  -p 8000:8000 \
  -e MT_WEB_HOST=0.0.0.0 \
  -e MT_WEB_PORT=8000 \
  -e MANGA_TRANSLATOR_ADMIN_PASSWORD=your_secure_password \
  -v $(pwd)/fonts:/app/fonts \
  -v $(pwd)/dict:/app/dict \
  -v $(pwd)/result:/app/result \
  your-registry/manga-translator:latest
```

#### GPU版本

```bash
# 拉取GPU镜像
docker pull your-registry/manga-translator:latest-gpu

# 运行GPU容器（带管理员密码）
docker run -d \
  --name manga-translator-gpu \
  --gpus all \
  -p 8000:8000 \
  -e MT_WEB_HOST=0.0.0.0 \
  -e MT_WEB_PORT=8000 \
  -e MT_USE_GPU=true \
  -e MT_MODELS_TTL=300 \
  -e MANGA_TRANSLATOR_ADMIN_PASSWORD=your_secure_password \
  -v $(pwd)/fonts:/app/fonts \
  -v $(pwd)/dict:/app/dict \
  -v $(pwd)/result:/app/result \
  your-registry/manga-translator:latest-gpu
```

### 方式2：使用Docker Compose

创建 `docker-compose.yml` 文件：

#### CPU版本

```yaml
version: '3.8'

services:
  manga-translator:
    image: your-registry/manga-translator:latest
    container_name: manga-translator
    ports:
      - "8000:8000"
    environment:
      - MT_WEB_HOST=0.0.0.0
      - MT_WEB_PORT=8000
      - MT_MODELS_TTL=300
      - MT_RETRY_ATTEMPTS=3
    volumes:
      - ./fonts:/app/fonts
      - ./dict:/app/dict
      - ./result:/app/result
      - ./models:/app/models
    restart: unless-stopped
```

#### GPU版本

```yaml
version: '3.8'

services:
  manga-translator-gpu:
    image: your-registry/manga-translator:latest-gpu
    container_name: manga-translator-gpu
    ports:
      - "8000:8000"
    environment:
      - MT_WEB_HOST=0.0.0.0
      - MT_WEB_PORT=8000
      - MT_USE_GPU=true
      - MT_MODELS_TTL=300
      - MT_RETRY_ATTEMPTS=3
      - MT_VERBOSE=false
    volumes:
      - ./fonts:/app/fonts
      - ./dict:/app/dict
      - ./result:/app/result
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

启动服务：

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

### 访问服务

容器启动后，通过浏览器访问：

- **用户界面**：`http://localhost:8000/`
- **管理后台**：`http://localhost:8000/admin`
- **API文档**：`http://localhost:8000/docs`

### 数据持久化

建议挂载以下目录以保持数据持久化：

- `/app/fonts` - 自定义字体文件
- `/app/dict` - 提示词和词典文件
- `/app/result` - 翻译结果输出
- `/app/models` - AI模型文件（可选，避免重复下载）
- `/app/.env` - 环境变量配置文件（可选）

### 配置API密钥

有两种方式配置API密钥：

#### 方式1：通过环境变量

在 `docker-compose.yml` 中添加：

```yaml
environment:
  - OPENAI_API_KEY=your_openai_key
  - GEMINI_API_KEY=your_gemini_key
  - DEEPL_AUTH_KEY=your_deepl_key
```

#### 方式2：通过.env文件

创建 `.env` 文件并挂载：

```bash
# .env 文件内容
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
DEEPL_AUTH_KEY=your_deepl_key
```

在 `docker-compose.yml` 中挂载：

```yaml
volumes:
  - ./.env:/app/.env
```

### 性能优化建议

1. **GPU内存管理**：
   ```yaml
   environment:
     - MT_MODELS_TTL=300  # 5分钟后卸载模型，节省显存
   ```

2. **限制容器资源**：
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 8G
   ```

3. **使用SSD存储**：
   - 将模型文件存储在SSD上以提高加载速度

### 故障排除

#### GPU不可用

**问题**：容器无法使用GPU

**解决方法**：
1. 确认已安装 NVIDIA Container Toolkit：
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```
2. 检查Docker版本是否支持GPU
3. 重启Docker服务

#### 端口冲突

**问题**：端口8000已被占用

**解决方法**：
修改端口映射：
```yaml
ports:
  - "8080:8000"  # 使用8080端口
environment:
  - MT_WEB_PORT=8000  # 容器内部仍使用8000
```

#### 模型下载缓慢

**问题**：首次启动时模型下载很慢

**解决方法**：
1. 预先下载模型文件到 `./models` 目录
2. 挂载模型目录：`-v $(pwd)/models:/app/models`

---

## 首次运行

### 1. 启动程序

双击 `app.exe`，程序会自动：
- 加载 AI 模型（首次运行需要几分钟）
- 初始化翻译引擎
- 打开主界面

### 2. 基础设置（CPU 版本用户必看）

如果使用 **CPU 版本**，请务必：

1. 点击"基础设置"标签页
2. **取消勾选"使用 GPU"**
3. 点击"保存配置"

> ⚠️ **重要**：CPU 版本如果启用 GPU 会导致程序崩溃！

### 3. 设置输出目录

1. 在主界面点击"选择输出文件夹"按钮
2. 选择翻译结果的保存位置
3. 程序会记住此设置

### 4. 选择翻译器

1. 在"基础设置"中找到"翻译器"下拉菜单
2. 首次使用推荐选择：
   - **高质量翻译 OpenAI** 或 **高质量翻译 Gemini**（多模态，看图翻译，效果最好）⭐ 强烈推荐
   - 需要配置 API Key → [查看 API 配置教程](API_CONFIG.md)

### 5. 添加图片

支持以下方式添加图片：

- **方式 1**：点击"添加文件"按钮选择图片
- **方式 2**：点击"添加文件夹"按钮选择文件夹
- **方式 3**：直接拖拽图片到窗口

支持的图片格式：`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

### 6. 开始翻译

1. 确认设置无误
2. 点击"开始翻译"按钮
3. 等待翻译完成
4. 结果会自动保存到输出文件夹

---

## 故障排除

### 程序无法启动

**问题**：双击 `app.exe` 没有反应或闪退

**解决方法**：
1. 检查是否解压了所有文件（不要直接在压缩包中运行）
2. 检查杀毒软件是否拦截了程序
3. 以管理员身份运行 `app.exe`
4. 查看 `logs/error.log` 文件

### 缺少 DLL 文件

**问题**：提示缺少 `VCRUNTIME140.dll` 或其他 DLL 文件

**解决方法**：
1. 下载并安装 [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. 重启电脑
3. 重新运行程序

### GPU 版本崩溃

**问题**：GPU 版本运行时崩溃或报错

**解决方法**：
1. 确认显卡支持 CUDA 12.x
2. 安装或更新 NVIDIA 显卡驱动
3. 下载并安装 [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
4. 如果仍然失败，使用 CPU 版本

### 翻译失败

**问题**：添加图片后翻译失败

**解决方法**：
1. 检查图片格式是否支持
2. 确认 `models/` 目录中的模型文件完整
3. 在"基础设置"中勾选"详细日志"查看错误信息
4. 查看 `logs/app.log` 文件

### 模型加载缓慢

**问题**：首次运行时模型加载时间过长

**原因**：程序需要加载多个 AI 模型文件（总计约 2-3 GB）

**建议**：
- 首次运行耐心等待 5-10 分钟
- 后续运行会快很多（模型已缓存）
- 建议安装在 SSD 上以提高加载速度

---

## 下一步

安装完成后，建议阅读以下文档：

- [功能特性](FEATURES.md) - 了解程序的所有功能
- [工作流程](WORKFLOWS.md) - 学习不同的翻译工作流程
- [设置说明](SETTINGS.md) - 配置翻译器和参数

---

返回 [主页](../README.md)

