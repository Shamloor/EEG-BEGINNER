"""
数据下载模块
用法：python download_data.py  # 单独测试下载
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
from config import IS_CLOUD, CLOUD_DATASET_URL, CLOUD_DATA_PATH, DATA_PATH


def download_file(url, local_path):
    """流式下载大文件，显示进度"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    print(f"Downloading from {url}")
    print(f"Save to: {local_path}")

    with open(local_path, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded / total_size * 100
                print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size})", end="")
    print("\nDownload completed!")


def extract_zip(zip_path, extract_to):
    """解压zip文件"""
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed!")


def ensure_data():
    """确保数据存在，如果不存在则下载"""
    # 检查数据是否已存在
    if os.path.exists(DATA_PATH) and os.listdir(DATA_PATH):
        print(f"Data already exists at {DATA_PATH}")
        return DATA_PATH

    if not IS_CLOUD:
        print("Local mode: Please ensure data is placed at", DATA_PATH)
        return DATA_PATH

    # 云模式：下载数据
    print("Cloud mode: Downloading dataset...")
    os.makedirs(CLOUD_DATA_PATH, exist_ok=True)

    # 下载临时文件
    zip_tmp = CLOUD_DATA_PATH + ".zip"
    try:
        download_file(CLOUD_DATASET_URL, zip_tmp)
        extract_zip(zip_tmp, CLOUD_DATA_PATH)
    finally:
        # 清理压缩包
        if os.path.exists(zip_tmp):
            os.remove(zip_tmp)

    # 可选：处理嵌套目录（如果zip里有多余的顶层文件夹）
    # 检查是否解压后多了一层，比如 data/hms-dataset/实际文件
    items = os.listdir(CLOUD_DATA_PATH)
    if len(items) == 1 and os.path.isdir(os.path.join(CLOUD_DATA_PATH, items[0])):
        inner_dir = os.path.join(CLOUD_DATA_PATH, items[0])
        for f in os.listdir(inner_dir):
            shutil.move(os.path.join(inner_dir, f), CLOUD_DATA_PATH)
        os.rmdir(inner_dir)

    print(f"Data ready at {DATA_PATH}")
    return DATA_PATH


def create_output_dirs():
    """创建输出目录"""
    if IS_CLOUD:
        cache_dir = os.path.join(CLOUD_DATA_PATH, "cache")
        output_dir = os.path.join(CLOUD_DATA_PATH, "outputs")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base_dir, "cache")
        output_dir = os.path.join(base_dir, "outputs")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directories created - cache: {cache_dir}, output: {output_dir}")


if __name__ == "__main__":
    # 单独测试下载功能
    ensure_data()
    create_output_dirs()
    print("Setup completed!")
