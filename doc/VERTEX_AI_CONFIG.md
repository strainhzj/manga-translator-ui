# Google Vertex AI API 配置指南

## ⚠️ 重要：认证错误诊断

如果您遇到以下错误：
```
401 UNAUTHENTICATED
ACCESS_TOKEN_TYPE_UNSUPPORTED
```

这通常表示：**API Key 格式不正确或使用了错误的认证方式**。

---

## 问题诊断

### 检查您的 API Key 格式

**❌ 错误的格式**：
```
AQ.Ab8RN6JRRqOgechiqF3TmWMCuUwc1BD2L2aO1ukgob_UobHQ2g
```
- 这看起来像 OAuth access token，不是 API Key
- Vertex AI 不支持这种认证方式

**✅ 正确的 Google Cloud API Key 格式**：
```
AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe
```
- 以 `AIza` 开头
- 约 39-40 字符长度
- 只包含字母和数字

---

## 解决方案

### 方案 1：使用 Google Cloud API Key（推荐）

#### 步骤 1：创建 Google Cloud 项目

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 登录您的 Google 账号
3. 点击顶部的项目选择器
4. 点击"新建项目"
5. 输入项目名称（如"Manga Translator"）
6. 点击"创建"

#### 步骤 2：启用 Vertex AI API

1. 在 Google Cloud Console 中，确保选择了刚创建的项目
2. 在搜索框中搜索"Vertex AI API"
3. 点击"Vertex AI API"
4. 如果提示启用，点击"启用"按钮

#### 步骤 3：创建 API Key

1. 在 Google Cloud Console 中，导航到：
   **APIs & Services** → **Credentials**（凭据）
2. 点击顶部的 **+ Create Credentials**（创建凭据）
3. 选择 **API key**（API 密钥）
4. 系统会自动生成一个 API Key，格式类似：
   ```
   AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe
   ```

#### 步骤 4：配置 API Key 限制（可选但推荐）

1. 点击刚创建的 API Key
2. 在"Application restrictions"（应用限制）中：
   - 选择"None"（无限制）或"IP addresses"（IP 地址）
   - 如果选择 IP 地址，添加您的公网 IP
3. 在"API restrictions"（API 限制）中：
   - 选择"Restrict key"（限制密钥）
   - 搜索并选择"Vertex AI API"
4. 点击"Save"（保存）

#### 步骤 5：配置到程序中

1. 打开程序
2. 在"基础设置"→"翻译器"中选择"Google Vertex AI"
3. 在"高级设置"中填写：
   - **API Key**：粘贴您的 Google Cloud API Key
   - **模型**：选择以下模型之一：
     - `gemini-2.5-flash-lite-preview-06-17`：速度快 ⭐ 推荐
     - `gemini-2.5-flash-preview-06-17`：平衡性能
     - `gemini-2.5-pro-preview-03-25`：质量最高
     - `gemini-1.5-flash-002`：稳定版本
     - `gemini-1.5-pro-002`：稳定版本

#### 步骤 6：验证配置

1. 点击"测试"按钮
2. 如果显示"连接成功"，说明配置正确
3. 如果仍然失败，检查：
   - API Key 是否完整复制
   - Google Cloud 项目是否启用了 Vertex AI API
   - 是否需要配置代理（见下方）

---

### 方案 2：使用 Gemini 翻译器（更简单）

如果您觉得 Vertex AI 配置复杂，可以使用 **Gemini 翻译器**，它使用 Google AI Studio API，配置更简单：

1. 访问 [Google AI Studio](https://aistudio.google.com/apikey)
2. 创建 API Key
3. 在程序中选择：
   - "高质量翻译 Gemini"（支持图片，质量更高）
   - 或 "Gemini"（仅文本，速度更快）
4. 填入 API Key 即可

详细配置方法见 [API_CONFIG.md](API_CONFIG.md)

---

## 代理配置（可选）

如果您的网络无法直接访问 Google 服务，需要配置代理：

### 配置方法

在项目根目录的 `.env` 文件中添加：

```env
# HTTPS 代理（推荐）
HTTPS_PROXY=http://127.0.0.1:7890

# HTTP 代理（备选）
HTTP_PROXY=http://127.0.0.1:7890
```

### 验证代理

启动程序后，查看日志中是否出现：
```
使用代理: http://127.0.0.1:7890
```

详细代理配置见 [PROXY_CONFIG.md](PROXY_CONFIG.md)

---

## 常见问题

### Q1：为什么要使用 Google Cloud API Key？

**答**：Vertex AI 是 Google Cloud 的企业级 AI 服务，需要使用 Google Cloud Console 创建的 API Key，而不是 Google AI Studio 的 API Key。

### Q2：API Key 和 OAuth Token 有什么区别？

**答**：
- **API Key**：简单的字符串验证，用于一般 API 调用
- **OAuth Token**：复杂的用户授权流程，用于访问用户数据

Vertex AI 的 REST API 端点只支持 API Key 认证。

### Q3：如何查看我的 API Key 类型？

**答**：
- Google Cloud API Key：以 `AIza` 开头
- OAuth Access Token：通常包含点号 `.` 或其他特殊字符

### Q4：配置后仍然报 401 错误？

**答**：检查以下项目：
1. ✅ API Key 是否完整复制（不要有空格或换行）
2. ✅ Google Cloud 项目是否启用了 Vertex AI API
3. ✅ API Key 是否有 API 限制（确保包含 Vertex AI API）
4. ✅ 网络是否需要代理配置
5. ✅ 模型名称是否正确

### Q5：Vertex AI 和 Gemini 翻译器有什么区别？

**答**：

| 特性 | Vertex AI | Gemini |
|------|-----------|--------|
| API 来源 | Google Cloud Console | Google AI Studio |
| 配置难度 | 较复杂 | 简单 |
| 功能 | 纯文本翻译 | 支持多模态（图片） |
| 价格 | 按 token 计费 | 按 token 计费 |
| 推荐 | 已有 Google Cloud 项目 | 新用户推荐 |

---

## 费用说明

- Vertex AI API 按使用量计费
- 新用户通常有免费额度
- 详细价格见：[Google Cloud Vertex AI 定价](https://cloud.google.com/vertex-ai/pricing)

---

## 技术细节

### API 端点

```
https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?key={API_KEY}
```

### 支持的模型

- `gemini-2.5-flash-lite-preview-06-17`
- `gemini-2.5-flash-preview-06-17`
- `gemini-2.5-pro-preview-03-25`
- `gemini-1.5-flash-002`
- `gemini-1.5-pro-002`
- `gemini-1.0-pro-vision`（多模态）

### 实现位置

- 翻译器代码：`manga_translator/translators/vertex.py`
- 配置模型：`manga_translator/translators/keys.py`

---

## 相关文档

- [代理配置指南](PROXY_CONFIG.md)
- [API 配置教程](API_CONFIG.md)
- [使用教程](USAGE.md)
- [Google Cloud 文档](https://cloud.google.com/vertex-ai/docs)

---

## 更新日志

- **2026-01-18**: 初始版本，添加 Vertex AI 配置指南
