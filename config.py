import os
import kagglehub

# ============ 运行环境控制 ============
IS_CLOUD = os.environ.get("IS_CLOUD", "false").lower() == "true"

# ============ 基础路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置（云模式 vs 本地模式）
if IS_CLOUD:
    # 云模式：使用 kagglehub 下载
    CLOUD_DATA_PATH = kagglehub.competition_download(
        "hms-harmful-brain-activity-classification"
    )
    DATA_PATH = CLOUD_DATA_PATH
else:
    # 本地模式
    DATA_PATH = os.path.join(
        BASE_DIR, "Data", "hms-harmful-brain-activity-classification"
    )

# 通用子目录配置
SPEC_PATH = os.path.join(DATA_PATH, "train_spectrograms")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 确保目录存在
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 通用文件路径配置
CACHE_SPECS_H5 = os.path.join(CACHE_DIR, "specs.h5")
OUTPUT_PKL_PATH = os.path.join(OUTPUT_DIR, "train_with_features.pkl")
XGBOOST_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_model.kpl")
XGBOOST_PRED_PATH = os.path.join(OUTPUT_DIR, "xgboost_predictions.csv")
CATBOOST_MODEL_PATH = os.path.join(OUTPUT_DIR, "catboost_model.kpl")
CATBOOST_PRED_PATH = os.path.join(OUTPUT_DIR, "catboost_predictions.csv")

# ============ 模型与训练配置 ============
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
MODEL = "CatBoost"
TEST_MODE = True