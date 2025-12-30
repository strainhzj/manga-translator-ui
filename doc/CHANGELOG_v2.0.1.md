# v2.0.1 更新日志

发布日期：2024-12-30

## 🐛 修复

### PaddleOCR 旋转裁剪尺寸限制修复
- 修复 PaddleOCR 韩语模型在处理超大文本区域时的崩溃问题
- 当文本区域透视变换目标尺寸超过 32767 像素时,采用局部裁剪策略
- 先从原图裁剪包围框区域,再进行透视变换,避免触发 OpenCV `cv::remap` 的 `SHRT_MAX` 限制
- 解决了大量 "Failed to extract rotated crop" 错误
