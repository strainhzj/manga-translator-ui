# Gemini 修复与重构计划

本文档记录了为解决项目构建、打包及自动更新问题而设计的最终修复方案。

## 第一部分：修改 GitHub Actions 工作流 (`.github/workflows/build-and-release.yml`)

目标：放弃 `gh-pages`，完全转向使用 GitHub Release 作为元数据和软件包的发布渠道，并支持大于 2GB 的文件和未来的差量更新。

### 1.1 修改 `build-cpu` 和 `build-gpu` Job

- **增加 7-Zip 安装**：在 `Install dependencies` 步骤中，通过 `choco install 7zip` 命令安装 7-Zip。
- **复制 7z.exe**：在 `Build ... version` 步骤中，增加一条命令 `cp 'C:\Program Files\7-Zip\7z.exe' .`，将 7-Zip 的可执行文件复制到项目根目录，以便 PyInstaller 可以打包它。

### 1.2 重构 `release-and-publish` Job

- **删除 `Deploy to GitHub Pages` 步骤**：完全移除此步骤。
- **增加 `Restore Previous TUF Assets` 步骤**：在 job 的最开始，此步骤会从上一个 GitHub Release 下载并合并分卷文件，为 `tufup` 创建差量包准备基础文件。
- **保留 `Restore TUF Private Keys` 步骤**：从 GitHub Secrets 中恢复签名私钥。
- **保留 `Update TUF Repository` 步骤**：此步骤现在会正确地生成所有元数据和更新包（包括大的 GPU 包）。
- **保留 `Bundle extra directories into __pycache__` 步骤**：确保您的额外数据文件被正确打包。
- **用 `Prepare All Release Assets` 替换旧的打包步骤**：这个统一的步骤将负责：
    a. 为手动下载创建 `.7z` 分卷包。
    b. 将 `tufup` 生成的所有元数据文件 (`*.json`) 复制到发布目录。
    c. 将 `tufup` 生成的小于 1.9GB 的更新包 (如 `cpu.tar.gz`) 直接复制到发布目录。
    d. 将 `tufup` 生成的大于 1.9GB 的更新包 (如 `gpu.tar.gz`) 分割成多个 `.7z` 分卷包。
- **保留 `Create GitHub Release` 步骤**：将 `release_assets` 目录下的所有文件（包括元数据、小包、分卷包）全部上传。

## 第二部分：修改客户端更新器 (`desktop-ui/updater.py`)

目标：使客户端能够从 GitHub Release 下载更新，并具备处理分卷包的能力。

- **重写 `main` 函数**：引入新的下载和更新逻辑。
- **修改 URL**：将 `metadata_base_url` 和 `target_base_url` 都指向 GitHub Release 的 `latest` 下载地址 (`https://github.com/hgmzhn/manga-translator-ui/releases/latest/download/`)。
- **增加分卷处理逻辑**：
    a. 在下载前，通过元数据判断目标文件是否为分卷包。
    b. 如果是分卷包，则按顺序下载所有分卷文件 (`.7z.001`, `.002` ...)。
    c. 调用打包进程序的 `7z.exe`，在本地临时目录中将分卷合并成一个完整的 `.tar.gz` 文件。
    d. 对合并后的文件进行哈希校验，然后解压安装。
- **增加 `requests` 和 `tqdm` 依赖**：用于实现带进度条的下载功能。

## 第三部分：修改 PyInstaller 打包配置 (`.spec` 文件)

目标：将 `7z.exe` 打包进最终的客户端程序中。

- **修改 `manga-translator-cpu.spec` 和 `manga-translator-gpu.spec`**：
- 在 `datas` 列表中，添加一条 `('7z.exe', '.')`，以确保 `7z.exe` 文件被包含在最终的发布包的根目录。
