import requests
import tarfile
import os
import shutil

def download_and_extract(url, target_dir):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 从URL中获取文件名
    filename = url.split('/')[-1]
    tar_path = os.path.join(target_dir, filename)
    
    # 下载文件
    print(f"正在下载 {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"{filename} 下载完成。")
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return

    # 解压文件
    print(f"正在解压 {filename}...")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)
        print(f"{filename} 解压完成。")
    except tarfile.TarError as e:
        print(f"解压失败: {e}")
        return
    finally:
        # 删除下载的压缩包
        if os.path.exists(tar_path):
            os.remove(tar_path)

if __name__ == '__main__':
    # 定义模型URL和目标目录
    cls_model_url = 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
    rec_model_url = 'https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_rec_server_infer.tar'
    output_dir = 'C:\\Users\\徐浩文\\manga-image-translator\\manga-translator-ui-package\\models\\ocr'

    # 下载并解压方向分类模型
    download_and_extract(cls_model_url, output_dir)
    # 下载并解压v5识别模型
    download_and_extract(rec_model_url, output_dir)

    # PaddleOCR的模型文件夹名通常是`ch_ppocr_mobile_v2.0_cls_infer`
    # 我们需要把它重命名为我们配置文件里期望的名字 `ch_ppocr_mobile_v2.0_cls`
    original_cls_dir = os.path.join(output_dir, 'ch_ppocr_mobile_v2.0_cls_infer')
    target_cls_dir = os.path.join(output_dir, 'ch_ppocr_mobile_v2.0_cls')
    if os.path.exists(original_cls_dir) and not os.path.exists(target_cls_dir):
        print(f"重命名 {original_cls_dir} 为 {target_cls_dir}")
        os.rename(original_cls_dir, target_cls_dir)

    # 同理，重命名识别模型文件夹
    original_rec_dir = os.path.join(output_dir, 'ch_PP-OCRv5_rec_server_infer')
    target_rec_dir = os.path.join(output_dir, 'PP-OCRv5_server_rec')
    if os.path.exists(original_rec_dir) and not os.path.exists(target_rec_dir):
        print(f"重命名 {original_rec_dir} 为 {target_rec_dir}")
        os.rename(original_rec_dir, target_rec_dir)

    print("所有操作完成。")