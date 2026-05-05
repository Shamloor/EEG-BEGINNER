# src/generate_npy.py
import pandas as pd
import numpy as np
import os
from config import SPEC_PATH, CACHE_SPECS_NPY, IS_CLOUD
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time


def load_single_parquet(file_name, spec_path):
    """
    读取单个parquet文件并转换为numpy数组

    Args:
        file_name: parquet文件名
        spec_path: 频谱图文件所在路径

    Returns:
        tuple: (sid, spectrogram_data) 或 (None, None) 如果读取失败
    """
    file_path = os.path.join(spec_path, file_name)
    try:
        df = pd.read_parquet(file_path)
        sid = int(file_name.split(".")[0])
        # 去掉时间列，只保留频谱数据
        spec_data = df.iloc[:, 1:].values
        return sid, spec_data
    except Exception as e:
        print(f"❌ 读取文件 {file_name} 时出错: {e}")
        return None, None


def generate_spectrograms_npy(use_multithreading=True, max_workers=None):
    """
    把所有频谱图 parquet 转换成一个 npy 文件
    只需要运行一次！

    参数:
        use_multithreading: 是否使用多线程（默认True）
        max_workers: 最大线程数（默认None，自动设置为CPU核心数）

    如果缓存文件已存在，会询问是否覆盖
    """

    print("开始生成频谱图缓存文件 specs.npy")
    start_time = time.time()

    # 1. 检查缓存文件是否已存在
    if os.path.exists(CACHE_SPECS_NPY):
        response = input(f"缓存文件已存在: {CACHE_SPECS_NPY}\n是否覆盖？(y/N): ")
        if response.lower() != "y":
            print("跳过生成，使用已有缓存文件")
            return
        print("覆盖已有缓存文件...")

    # 2. 检查原始数据路径是否存在
    if not os.path.exists(SPEC_PATH):
        print(f"❌ 错误：频谱图路径不存在: {SPEC_PATH}")
        if IS_CLOUD:
            print("云环境下请检查数据是否正确下载，或 CACHE_SPECS_NPY 是否已存在")
        else:
            print("请确认本地数据路径配置正确")
        return

    # 3. 获取所有parquet文件
    files = [f for f in os.listdir(SPEC_PATH) if f.endswith(".parquet")]
    total = len(files)

    if total == 0:
        print(f"❌ 错误：{SPEC_PATH} 下没有找到 .parquet 文件")
        return

    print(f"共找到 {total} 个频谱图文件")

    # 4. 选择读取方式
    spectrograms = {}

    if use_multithreading and total > 10:  # 文件数量少时没必要用多线程
        # 设置线程数
        if max_workers is None:
            # I/O密集型任务可以设置比CPU核心数更大的值
            cpu_count = multiprocessing.cpu_count()
            # Kaggle CPU环境有4核，可以设置8-16个线程
            max_workers = min(cpu_count * 2, 32)
            print(f"检测到CPU核心数: {cpu_count}")

        print(f"🚀 使用多线程模式，线程数: {max_workers}")

        # 使用线程池并行读取
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(load_single_parquet, f, SPEC_PATH): f for f in files
            }

            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_file):
                sid, data = future.result()
                if sid is not None:
                    spectrograms[sid] = data

                completed += 1
                # 打印进度
                if completed % 100 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"进度: {completed}/{total} ({completed/total*100:.1f}%) "
                        f"- 速度: {speed:.1f} 文件/秒",
                        end="\r",
                    )

        print()  # 换行
    else:
        # 单线程模式（原始方式）
        if total <= 10:
            print("📀 文件数量较少，使用单线程模式")
        else:
            print("📀 使用单线程模式（可通过设置 use_multithreading=True 启用多线程）")

        for i, f in enumerate(files):
            if (i + 1) % 100 == 0 or i + 1 == total:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"进度: {i+1}/{total} ({speed:.1f} 文件/秒)", end="\r")

            path = os.path.join(SPEC_PATH, f)
            df = pd.read_parquet(path)
            sid = int(f.split(".")[0])
            spectrograms[sid] = df.iloc[:, 1:].values

        print()  # 换行

    # 5. 保存为npy文件
    print(f"正在保存到 {CACHE_SPECS_NPY}...")
    save_path = CACHE_SPECS_NPY
    np.save(save_path, spectrograms)

    # 6. 输出统计信息
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(save_path) / (1024**3)  # 转换为GB

    print(f"\n✅ 成功生成 specs.npy → {save_path}")
    print(f"   - 包含 {len(spectrograms)} 个频谱图")
    print(f"   - 文件大小: {file_size:.2f} GB")
    print(f"   - 总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"   - 平均速度: {total/elapsed_time:.2f} 文件/秒")
