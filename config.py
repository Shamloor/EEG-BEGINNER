import os
import kagglehub

# ============ 运行环境控制 ============
# 通过环境变量 IS_CLOUD=true 切换到云模式
IS_CLOUD = os.environ.get("IS_CLOUD", "false").lower() == "true"

# ============ 数据源配置 ============
# 云模式下的数据集下载（使用kagglehub）
if IS_CLOUD:
    # 使用kagglehub下载竞赛数据
    CLOUD_DATASET_PATH = kagglehub.competition_download(
        "hms-harmful-brain-activity-classification"
    )
else:
    CLOUD_DATASET_PATH = None

# 本地模式的数据路径
LOCAL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Data",
    "hms-harmful-brain-activity-classification",
)

# 云模式的数据路径（临时/持久化目录）
CLOUD_DATA_PATH = os.environ.get("CLOUD_DATA_PATH", "/home/ubuntu/data/hms-dataset")

# 统一数据路径（代码中统一使用这个变量）
if IS_CLOUD:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = CLOUD_DATA_PATH

    # 定义目录
    CACHE_DIR = os.path.join(CLOUD_DATA_PATH, "cache")
    OUTPUT_DIR = os.path.join(CLOUD_DATA_PATH, "outputs")

    # 路径配置
    CACHE_SPECS_NPY = os.path.join(CACHE_DIR, "specs.npy")
    OUTPUT_PKL_PATH = os.path.join(OUTPUT_DIR, "train_with_features.pkl")
    XGB_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_model.kpl")
    XGB_PRED_PATH = os.path.join(OUTPUT_DIR, "xgboost_predictions.csv")

else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = LOCAL_DATA_PATH
    SPEC_PATH = os.path.join(DATA_PATH, "train_spectrograms")

    # 本地也使用 cache/ 和 outputs/ 分离的结构
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    CACHE_SPECS_NPY = os.path.join(CACHE_DIR, "specs.npy")
    OUTPUT_PKL_PATH = os.path.join(OUTPUT_DIR, "train_with_features.pkl")
    XGB_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_model.kpl")
    XGB_PRED_PATH = os.path.join(OUTPUT_DIR, "xgboost_predictions.csv")

# 确保输出目录存在
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 以下配置保持不变 ============
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

USE_NPY_CACHE = True
FEATURE_ENGINEER = True
MODEL = "XGBoost"
