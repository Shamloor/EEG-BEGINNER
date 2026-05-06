# src/generate_h5.py
import pandas as pd
import numpy as np
import h5py
import os
from config import SPEC_PATH, CACHE_SPECS_H5, IS_CLOUD
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time
from threading import Lock


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
        spec_data = df.iloc[:, 1:].values.astype(np.float32)
        return sid, spec_data
    except Exception as e:
        print(f"❌ 读取文件 {file_name} 时出错: {e}")
        return None, None


def write_batch_to_h5(h5_path, batch_data, lock):
    """
    将一批数据写入HDF5文件

    Args:
        h5_path: HDF5文件路径
        batch_data: dict {sid: spectrogram_data}
        lock: 线程锁，保证写入安全
    """
    with lock:
        with h5py.File(h5_path, "a") as f:
            for sid, data in batch_data.items():
                # 将sid转为字符串作为dataset名称（HDF5不支持纯数字作为名称）
                key = str(sid)
                f.create_dataset(key, data=data, compression="gzip", compression_opts=4)


def generate_spectrograms_h5(use_multithreading=True, max_workers=None, batch_size=100):
    """
    把所有频谱图 parquet 转换成一个 HDF5 文件（分批写入，控制内存）

    Args:
        use_multithreading: 是否使用多线程（默认True）
        max_workers: 最大线程数（默认None，自动设置为CPU核心数）
        batch_size: 每批写入的文件数量（默认100）
    """
    print("开始生成频谱图缓存文件 specs.h5")
    start_time = time.time()

    # 1. 检查缓存文件是否已存在
    if os.path.exists(CACHE_SPECS_H5):
        if IS_CLOUD:
            # 云环境：文件已存在，直接跳过
            print(f"云环境检测到缓存文件已存在，跳过生成: {CACHE_SPECS_H5}")
            return
        else:
            # 本地环境：询问用户
            response = input(f"缓存文件已存在: {CACHE_SPECS_H5}\n是否覆盖？(y/N): ")
            if response.lower() != "y":
                print("跳过生成，使用已有缓存文件")
                return
            print("覆盖已有缓存文件...")
            # 删除旧文件
            os.remove(CACHE_SPECS_H5)

    # 2. 检查原始数据路径是否存在
    if not os.path.exists(SPEC_PATH):
        print(f"❌ 错误：频谱图路径不存在: {SPEC_PATH}")
        if IS_CLOUD:
            print("云环境下请检查数据是否正确下载，或 CACHE_SPECS_H5 是否已存在")
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
    print(f"批量大小: {batch_size} 个文件/批")

    # 4. 分批读取并写入HDF5
    batch_buffer = {}
    batch_count = 0
    write_lock = Lock()

    # 预先创建HDF5文件（空文件）
    with h5py.File(CACHE_SPECS_H5, "w") as f:
        pass  # 只创建文件，不写入数据

    if use_multithreading and total > 10:
        # 设置线程数
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            max_workers = min(cpu_count * 2, 32)
            print(f"检测到CPU核心数: {cpu_count}")

        print(f"🚀 使用多线程模式，线程数: {max_workers}")

        # 使用线程池并行读取
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(load_single_parquet, f, SPEC_PATH): f for f in files
            }

            completed = 0
            for future in as_completed(future_to_file):
                sid, data = future.result()
                if sid is not None:
                    batch_buffer[sid] = data

                completed += 1

                # 当缓冲区达到批量大小时，写入HDF5并清空缓冲区
                if len(batch_buffer) >= batch_size:
                    batch_count += 1
                    print(
                        f"\n写入第 {batch_count} 批数据 ({len(batch_buffer)} 个文件)...",
                        end=" ",
                    )
                    write_batch_to_h5(CACHE_SPECS_H5, batch_buffer, write_lock)
                    batch_buffer.clear()  # 清空缓冲区，释放内存
                    print("完成")

                # 打印进度
                if completed % 100 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"进度: {completed}/{total} ({completed/total*100:.1f}%) "
                        f"- 速度: {speed:.1f} 文件/秒",
                        end="",
                    )

        # 写入最后一批不足batch_size的数据
        if batch_buffer:
            batch_count += 1
            print(
                f"\n写入第 {batch_count} 批数据 ({len(batch_buffer)} 个文件)...",
                end=" ",
            )
            write_batch_to_h5(CACHE_SPECS_H5, batch_buffer, write_lock)
            batch_buffer.clear()
            print("完成")

        print()  # 换行
    else:
        # 单线程模式
        if total <= 10:
            print("文件数量较少，使用单线程模式")
        else:
            print("使用单线程模式（可通过设置 use_multithreading=True 启用多线程）")

        for i, f in enumerate(files):
            if (i + 1) % 100 == 0 or i + 1 == total:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"进度: {i+1}/{total} ({speed:.1f} 文件/秒)", end="\r")

            path = os.path.join(SPEC_PATH, f)
            df = pd.read_parquet(path)
            sid = int(f.split(".")[0])
            spec_data = df.iloc[:, 1:].values.astype(np.float32)
            batch_buffer[sid] = spec_data

            # 达到批量大小时写入
            if len(batch_buffer) >= batch_size:
                batch_count += 1
                write_batch_to_h5(CACHE_SPECS_H5, batch_buffer, write_lock)
                batch_buffer.clear()

        # 写入最后一批
        if batch_buffer:
            batch_count += 1
            write_batch_to_h5(CACHE_SPECS_H5, batch_buffer, write_lock)
            batch_buffer.clear()

        print()  # 换行

    # 5. 输出统计信息
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(CACHE_SPECS_H5) / (1024**3)  # 转换为GB

    print(f"\n✅ 成功生成频谱图缓存文件 → {CACHE_SPECS_H5}")
    print(f"   - 包含 {total} 个频谱图")
    print(f"   - 写入批次: {batch_count}")
    print(f"   - 文件大小: {file_size:.2f} GB")
    print(f"   - 总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"   - 平均速度: {total/elapsed_time:.2f} 文件/秒")
