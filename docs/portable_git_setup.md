# 便携式 Git 配置

## 📦 为什么需要便携式 Git?

**BallonsTranslator-dev** 的做法是在项目中包含便携式Git,这样:
- ✅ 用户无需安装Git
- ✅ 自动更新功能开箱即用
- ✅ 版本统一,避免兼容性问题

## 🚀 快速设置

### 方法一: 使用下载脚本(推荐)

```bash
# 双击运行:
download_portable_git.bat
```

脚本会:
1. 从 GitHub 下载 Git 便携版 (约50MB)
2. 自动解压到 `PortableGit` 目录
3. 配置 Git 用户信息

### 方法二: 手动下载

1. **下载Git便携版**:
   - 访问: https://git-scm.com/download/win
   - 选择 "Portable ('thumbdrive edition')"
   - 下载 64-bit 版本

2. **解压到项目目录**:
   ```
   manga-translator-ui-package/
   ├── PortableGit/          # 解压到这里
   │   ├── cmd/
   │   │   └── git.exe
   │   ├── bin/
   │   └── ...
   ├── launch.py
   └── launch_win_with_autoupdate.bat
   ```

3. **配置Git**:
   ```bash
   PortableGit\cmd\git.exe config --global user.name "Your Name"
   PortableGit\cmd\git.exe config --global user.email "your@email.com"
   ```

## 📂 目录结构

```
manga-translator-ui-package/
├── PortableGit/              # 便携式Git (约150MB解压后)
│   ├── cmd/
│   │   ├── git.exe          # Git命令行工具
│   │   └── ...
│   ├── bin/
│   ├── mingw64/
│   └── ...
├── launch.py                 # 自动检测并使用便携版Git
├── launch_win_with_autoupdate.bat
└── download_portable_git.bat # 下载脚本
```

## 🔧 工作原理

### 自动检测逻辑

启动脚本会按以下优先级查找Git:

1. **便携版Git**: `PortableGit/cmd/git.exe`
2. **系统Git**: 从 PATH 环境变量查找
3. **自定义路径**: 通过 `GIT` 环境变量指定

```python
# launch.py 中的检测代码
portable_git = PATH_ROOT / "PortableGit" / "cmd" / "git.exe"
if portable_git.exists():
    git = str(portable_git)  # 优先使用便携版
else:
    git = os.environ.get('GIT', "git")  # 降级到系统Git
```

## ⚙️ 高级配置

### 使用系统Git而非便携版

如果你已安装系统Git,可以删除 `PortableGit` 目录,脚本会自动使用系统Git。

### 指定自定义Git路径

```bash
# Windows
set GIT=C:\Custom\Path\To\git.exe
launch_win_with_autoupdate.bat

# 或在 launch.py 中:
python launch.py --update
```

### 国内下载加速

如果GitHub下载太慢,可以使用镜像:

1. **修改 download_portable_git.bat** 中的下载地址为:
   ```
   https://npm.taobao.org/mirrors/git-for-windows/...
   ```

2. **或手动从国内镜像下载**:
   - 腾讯云: https://mirrors.cloud.tencent.com/github-release/git-for-windows/
   - 清华源: https://mirrors.tuna.tsinghua.edu.cn/

## 🎯 部署建议

### 开发版 (Source Code)
- ❌ 不包含 PortableGit
- 用户需要自己安装或下载

### 便携版 (Portable Release)
- ✅ 包含 PortableGit  
- 用户解压即用,无需安装
- 文件大小增加约 150MB

### 安装版 (Installer)
- ⚡ 可选组件: 在安装时询问是否安装Git
- 或检测系统是否已安装Git

## 📊 便携版优缺点

### ✅ 优点:
- 用户无需安装Git
- 版本统一,兼容性好
- 自动更新功能开箱即用
- 不污染系统环境

### ❌ 缺点:
- 项目体积增大约150MB
- 需要额外下载时间
- 占用磁盘空间

## 💡 建议

**对于你的项目,建议:**

1. **GitHub Release**: 不包含Git,提供下载脚本
2. **用户首次运行**: 
   - 检测到无Git → 提示运行 `download_portable_git.bat`
   - 或提供安装系统Git的链接

3. **完整版发布**: 可以制作包含Git的"完整便携版"

---

## 🔍 相关文件

- `download_portable_git.bat` - Git便携版下载脚本
- `launch.py` - 自动检测并使用便携版Git
- `launch_win_with_autoupdate.bat` - 使用Git自动更新

## 📚 参考链接

- Git Windows下载: https://git-scm.com/download/win
- Git便携版说明: https://git-scm.com/docs/git-for-windows
- 便携版下载直链: https://github.com/git-for-windows/git/releases

