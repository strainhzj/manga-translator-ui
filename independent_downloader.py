
import paddlex as pdx

if __name__ == '__main__':
    print("开始下载方向分类模型...")
    pdx.deploy.Predictor(model_dir=pdx.utils.download_and_decompress('https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'))
    print("方向分类模型下载完成。")
    
    print("开始下载v5识别模型...")
    pdx.deploy.Predictor(model_dir=pdx.utils.download_and_decompress('https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_rec_server_infer.tar'))
    print("v5识别模型下载完成。")
