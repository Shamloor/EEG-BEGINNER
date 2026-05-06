import numpy as np
import pandas as pd
import os
import h5py
import warnings

warnings.filterwarnings("ignore")

from config import SPEC_PATH, USE_NPY_CACHE, OUTPUT_PKL_PATH, IS_CLOUD


def check_and_handle_existing_pkl():
    """检查 pkl 文件是否存在，如果存在则询问用户是否覆盖"""
    if os.path.exists(OUTPUT_PKL_PATH):
        print(f"\n⚠️ 特征文件已存在: {OUTPUT_PKL_PATH}")

        if IS_CLOUD:
            # 云环境：文件已存在，直接跳过
            print("云环境检测到特征文件已存在，跳过生成")
            return False
        else:
            # 本地环境：询问用户
            response = input("是否覆盖重新生成？(y/N): ")
            if response.lower() != "y":
                print("跳过特征工程，使用已有特征文件")
                return False
            print("覆盖已有文件，重新生成特征...")
    return True


def load_spectrograms():
    if not USE_NPY_CACHE:
        # 方式A：加载所有单独的parquet文件（慢，但保证有数据）
        files = os.listdir(SPEC_PATH)
        print(f"   找到 {len(files)} 个频谱图parquet文件。开始加载...")
        spectrograms = {}

        for i, f in enumerate(files):
            if i % 100 == 0:
                print(f"   已加载 {i} 个文件...", end="\r")
            tmp = pd.read_parquet(os.path.join(SPEC_PATH, f))
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        print(f"   加载完成。共加载 {len(spectrograms)} 个频谱图。")

    else:
        # 方式B：加载预存的.h5文件（内存友好，按需读取）
        from config import CACHE_SPECS_H5

        try:
            spectrograms_path = CACHE_SPECS_H5
            # 只打开文件句柄，不加载数据到内存
            spectrograms = h5py.File(spectrograms_path, "r")
            print(
                f"   从 {spectrograms_path} 成功打开 HDF5 频谱图文件（数据按需读取）。"
            )
            print(f"   包含 {len(spectrograms)} 个频谱图。")
        except FileNotFoundError:
            print("错误: 未找到预存的 specs.h5 文件。")
            print("解决方案: 1) 将 READ_SPEC_FILES 设为 True 重新运行。")
            print("         2) 或确保 specs.h5 文件存在于指定路径。")
            spectrograms = None
        except Exception as e:
            print(f"错误: 打开 HDF5 文件失败: {e}")
            spectrograms = None

    return spectrograms


def extract_features(train_non_overlap, spectrograms):
    """
    从频谱图中提取 10 分钟 + 20 秒窗口特征

    Args:
        train_non_overlap: 频谱图的索引数据, 包括spec_id, min_time, max_time, patient_id
        spectrograms: 频谱图列表（NPY模式下是dict，HDF5模式下是h5py.File对象）
    """
    sample_spec = pd.read_parquet(os.path.join(SPEC_PATH, "1000086677.parquet"))
    SPEC_COLS = sample_spec.columns[1:]
    n_freqs = len(SPEC_COLS)

    FEATURES = []
    FEATURES += [f"{c}_mean_10m" for c in SPEC_COLS]
    FEATURES += [f"{c}_min_10m" for c in SPEC_COLS]
    FEATURES += [f"{c}_mean_20s" for c in SPEC_COLS]
    FEATURES += [f"{c}_min_20s" for c in SPEC_COLS]

    feature_data = np.zeros((len(train_non_overlap), len(FEATURES)))

    for k in range(len(train_non_overlap)):
        if k % 500 == 0:
            print(f"处理 {k}/{len(train_non_overlap)}", end="\r")

        row = train_non_overlap.iloc[k]
        r = int((row.min_time + row.max_time) // 4)

        # 直接从 HDF5 文件中获取 dataset , 注意类型转换
        spec = spectrograms.get(str(int(row.spec_id)))
        if spec is None:
            continue

        # 注意：HDF5 dataset 也支持 len() 和切片操作
        # 10 分钟窗口
        start = r
        end = r + 300
        end = min(end, len(spec))
        start = max(0, end - 300)
        win = spec[start:end]  # HDF5 模式下此处会从磁盘读取数据

        feature_data[k, :n_freqs] = np.nanmean(win, axis=0)
        feature_data[k, n_freqs : 2 * n_freqs] = np.nanmin(win, axis=0)

        # 20 秒窗口
        start = r + 145
        end = r + 155
        end = min(end, len(spec))
        start = max(0, end - 10)
        win = spec[start:end]  # HDF5 模式下此处会从磁盘读取数据

        feature_data[k, 2 * n_freqs : 3 * n_freqs] = np.nanmean(win, axis=0)
        feature_data[k, 3 * n_freqs : 4 * n_freqs] = np.nanmin(win, axis=0)

    features_df = pd.DataFrame(
        feature_data, columns=FEATURES, index=train_non_overlap.index
    )
    final_df = pd.concat([train_non_overlap, features_df], axis=1)
    return final_df


def save_features(df):
    """保存特征到 pkl 文件"""
    df.to_pickle(OUTPUT_PKL_PATH)
    print(f"✅ 特征已保存到：{OUTPUT_PKL_PATH}")


# ============ 主入口函数（供 main.py 调用） ============
def run_feature_engineering(train_non_overlap):
    """
    特征工程主入口
    包含：检查已有文件、加载频谱图、提取特征、保存
    """
    # 1. 检查是否已有特征文件
    if not check_and_handle_existing_pkl():
        return None

    # 2. 加载频谱图
    print("\n--- 加载频谱图 ---")
    spectrograms = load_spectrograms()
    if spectrograms is None:
        print("❌ 频谱图加载失败，特征工程终止")
        return None

    # 3. 提取特征
    print("\n--- 提取特征 ---")
    final_df = extract_features(train_non_overlap, spectrograms)

    # 如果是 HDF5 模式，关闭文件句柄
    if USE_NPY_CACHE and hasattr(spectrograms, "close"):
        spectrograms.close()
        print("已关闭 HDF5 文件句柄")

    # 4. 保存特征
    print("\n--- 保存特征 ---")
    save_features(final_df)

    return final_df
