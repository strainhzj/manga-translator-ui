
import requests
import tarfile
import os

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
            # 解压所有文件到目标目录的一个临时子文件夹
            temp_extract_dir = os.path.join(target_dir, 'temp_extract')
            tar.extractall(path=temp_extract_dir)
            # 移动文件到目标目录
            extracted_folder = os.path.join(temp_extract_dir, 'ch_ppocr_mobile_v2.0_cls_infer')
            for item in os.listdir(extracted_folder):
                s = os.path.join(extracted_folder, item)
                d = os.path.join(target_dir, item)
                os.rename(s, d)
            os.rmdir(extracted_folder)
            os.rmdir(temp_extract_dir)
        print(f"{filename} 解压完成。")
    except (tarfile.TarError, FileNotFoundError) as e:
        print(f"解压或移动文件失败: {e}")
    finally:
        # 删除下载的压缩包
        if os.path.exists(tar_path):
            os.remove(tar_path)

if __name__ == '__main__':
    cls_model_url = 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
    output_dir = 'C:\Users\徐浩文\manga-image-translator\manga-translator-ui-package\models\ocr\ch_ppocr_mobile_v2.0_cls'

    download_and_extract(cls_model_url, output_dir)

    print("方向分类模型下载操作完成。")
