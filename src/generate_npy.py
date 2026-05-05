# src/generate_npy.py
import pandas as pd
import numpy as np
import os
from config import SPEC_PATH, CACHE_SPECS_NPY, IS_CLOUD


def generate_spectrograms_npy():
    """
    把所有频谱图 parquet 转换成一个 npy 文件
    只需要运行一次！

    如果缓存文件已存在，会询问是否覆盖（云上默认跳过）
    """

    print("开始生成频谱图缓存文件 specs.npy")

    # 检查缓存文件是否已存在
    if os.path.exists(CACHE_SPECS_NPY):
        # 云上和本地都询问是否覆盖
        response = input(f"缓存文件已存在: {CACHE_SPECS_NPY}\n是否覆盖？(y/N): ")
        if response.lower() != "y":
            print("跳过生成，使用已有缓存文件")
            return
        print("覆盖已有缓存文件...")

    # 3. 检查原始数据路径是否存在
    if not os.path.exists(SPEC_PATH):
        print(f"❌ 错误：频谱图路径不存在: {SPEC_PATH}")
        if IS_CLOUD:
            print("云环境下请检查数据是否正确下载，或 CACHE_SPECS_NPY 是否已存在")
        else:
            print("请确认本地数据路径配置正确")
        return

    # 4. 读取所有 parquet 并保存到字典
    spectrograms = {}
    files = [f for f in os.listdir(SPEC_PATH) if f.endswith(".parquet")]
    total = len(files)

    if total == 0:
        print(f"❌ 错误：{SPEC_PATH} 下没有找到 .parquet 文件")
        return

    print(f"共找到 {total} 个频谱图文件")

    for i, f in enumerate(files):
        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"进度: {i+1}/{total}", end="\r")

        path = os.path.join(SPEC_PATH, f)
        df = pd.read_parquet(path)
        sid = int(f.split(".")[0])
        spectrograms[sid] = df.iloc[:, 1:].values  # 去掉时间列

    # 5. 保存为 npy
    save_path = CACHE_SPECS_NPY
    np.save(save_path, spectrograms)
    print(f"\n✅ 成功生成 specs.npy → {save_path}")
    print(f"包含 {len(spectrograms)} 个频谱图")
