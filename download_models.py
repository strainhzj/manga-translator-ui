from manga_translator.ocr.paddle_ocr.paddleocr import PaddleOCR
if __name__ == '__main__':
    print("开始下载方向分类模型...")
    cls_engine = PaddleOCR(use_angle_cls=True, lang='ch', ocr_version='PP-OCRv3', device='cpu')
    print("方向分类模型下载完成。")
    print("开始下载v5识别模型...")
    rec_engine = PaddleOCR(lang='ch', ocr_version='PP-OCRv5', device='cpu')
    print("v5识别模型下载完成。")