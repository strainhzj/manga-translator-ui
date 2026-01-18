# 代理配置指南

## 概述

本项目支持通过 HTTP/HTTPS 代理服务器访问需要网络的翻译器（如 Google Vertex AI、OpenAI 等）。

## 支持的翻译器

代理功能已应用于以下翻译器：
- ✅ **Google Vertex AI** (完全支持)
- ✅ **API 测试功能**（所有支持测试的翻译器）

## 配置方法

### 方式 1：环境变量（推荐）

在项目根目录的 `.env` 文件中添加代理配置：

```env
# HTTPS 代理（推荐用于 Vertex AI）
HTTPS_PROXY=http://127.0.0.1:7890

# HTTP 代理（作为备选）
HTTP_PROXY=http://127.0.0.1:7890
```

**优先级**：
- 对于 HTTPS 请求（如 Vertex AI），优先使用 `HTTPS_PROXY`
- 如果未设置 `HTTPS_PROXY`，则回退到 `HTTP_PROXY`

### 方式 2：系统环境变量

在操作系统中设置环境变量：

**Windows (PowerShell)**:
```powershell
$env:HTTPS_PROXY="http://127.0.0.1:7890"
$env:HTTP_PROXY="http://127.0.0.1:7890"
```

**Windows (CMD)**:
```cmd
set HTTPS_PROXY=http://127.0.0.1:7890
set HTTP_PROXY=http://127.0.0.1:7890
```

**Linux/macOS**:
```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
```

### 方式 3：代理配置格式

支持的代理格式：

```
# HTTP 代理
http://127.0.0.1:7890
http://localhost:7890

# 带域名的代理
http://proxy.example.com:8080

# HTTPS 代理
https://127.0.0.1:7890
```

## 常见代理软件配置示例

### Clash

**默认端口**: 7890

```env
HTTPS_PROXY=http://127.0.0.1:7890
```

### V2Ray / V2RayN

**默认端口**: 10809 (HTTP) 或 10808 (SOCKS)

⚠️ **注意**: 本项目仅支持 HTTP/HTTPS 代理，不支持 SOCKS 代理。

```env
# 使用 V2Ray 的 HTTP 代理端口
HTTPS_PROXY=http://127.0.0.1:10809
```

### Shadowsocks / SSR

如果使用 Shadowsocks，需要配合 Privoxy 等工具转换为 HTTP 代理。

### 其他代理软件

请查看您使用的代理软件的 HTTP 代理端口号，然后按照上述格式配置。

## 验证代理配置

### 方法 1：查看应用日志

启动应用后，如果代理配置成功，会在日志中看到：

```
使用代理: http://127.0.0.1:7890
```

### 方法 2：使用 API 测试功能

1. 在应用中选择 **"Google Vertex AI"** 翻译器
2. 配置 `VERTEX_API_KEY`
3. 点击 **"测试"** 按钮
4. 如果连接成功，说明代理配置正确

## 故障排查

### 问题 1：无法连接到代理服务器

**错误信息**: `Cannot connect to host ...`

**解决方案**:
1. 确认代理软件正在运行
2. 检查代理端口号是否正确
3. 尝试在浏览器中测试代理是否正常工作

### 问题 2：代理认证失败

**错误信息**: `407 Proxy Authentication Required`

**说明**: 当前版本不支持代理认证（用户名/密码）

**解决方案**:
1. 在代理软件中设置允许本地连接（白名单）
2. 使用不需要认证的代理端口

### 问题 3：连接超时

**错误信息**: `信号灯超时时间已到` 或 `Timeout`

**可能原因**:
1. 代理服务器响应慢
2. 网络连接不稳定
3. 代理服务器被防火墙阻止

**解决方案**:
1. 尝试更换代理服务器
2. 检查防火墙设置
3. 尝试直接连接（不使用代理）

### 问题 4：代理未生效

**检查步骤**:
1. 确认 `.env` 文件位于项目根目录
2. 确认环境变量名称正确（`HTTPS_PROXY` 或 `HTTP_PROXY`）
3. 重启应用使配置生效
4. 查看日志中是否出现"使用代理"的信息

## 安全建议

1. ⚠️ **不要提交 `.env` 文件到版本控制系统**
   - `.env` 文件已在 `.gitignore` 中
   - 代理配置可能包含敏感信息

2. 🔒 **使用可信的代理服务器**
   - 只使用您自己搭建或信任的代理
   - 不要在公共代理中传输敏感数据（API Key）

3. 📝 **定期检查代理配置**
   - 确保代理设置符合当前需求
   - 不使用时可以注释掉代理配置

## 其他注意事项

### 代理性能

使用代理可能会降低翻译速度，这是正常现象。如果发现速度过慢：
- 尝试更换代理服务器
- 选择延迟更低的代理节点
- 考虑使用直连（如果网络环境允许）

### 局域网代理

如果代理服务器在局域网内，使用局域网 IP 地址：

```env
HTTPS_PROXY=http://192.168.1.100:7890
```

### 仅对特定翻译器使用代理

当前实现中，代理配置对所有支持的网络请求生效。如果需要仅对特定翻译器使用代理，可以考虑：
- 使用多个 `.env` 文件并手动切换
- 使用环境变量管理工具（如 `direnv`）

## 相关文档

- [API 配置指南](API_CONFIG.md)
- [使用教程](USAGE.md)
- [调试指南](DEBUGGING.md)

## 技术细节

### 实现位置

- **翻译器**: `manga_translator/translators/vertex.py`
- **API 测试**: `desktop_qt_ui/app_logic.py`

### 依赖库

- `aiohttp`: 异步 HTTP 客户端（原生支持代理）

### 代理传递流程

```
.env 文件
  ↓
os.getenv() 读取环境变量
  ↓
aiohttp.ClientSession(proxy=proxy)
  ↓
HTTP 请求通过代理发送
```

## 更新日志

- **2026-01-17**: 初始版本，添加 Google Vertex AI 代理支持
