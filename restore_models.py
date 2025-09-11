
import requests
import os

def download_file(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    filename = url.split('/')[-1]
    file_path = os.path.join(target_dir, filename)
    
    if os.path.exists(file_path):
        print(f"{filename} 已存在，跳过下载。")
        return

    print(f"正在下载 {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"{filename} 下载完成。")
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")

if __name__ == '__main__':
    # 定义模型URL和目标目录
    rec_model_url = 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt'
    dict_url = 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt'
    output_dir = 'C:\Users\徐浩文\manga-image-translator\manga-translator-ui-package\models\ocr'

    # 下载并恢复文件
    download_file(rec_model_url, output_dir)
    download_file(dict_url, output_dir)

    print("恢复操作完成。")
