from download_data import ensure_data, create_output_dirs
from src.data_loader import load_train_csv
from src.preprocess import create_non_overlap_data
from src.generate_npy import generate_spectrograms_npy
from src.feature_engineering import run_feature_engineering
from src.train_xgboost import load_features, prepare_data, train_xgboost_with_group_cv
from config import FEATURE_ENGINEER, MODEL, IS_CLOUD


def print_section(title, step_num, total_steps=None):
    """打印格式化的章节标题"""
    if total_steps:
        print(f"\n{'=' * 60}")
        print(f"  [{step_num}/{total_steps}] {title}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'-' * 40}")
        print(f"  {title}")
        print(f"{'-' * 40}")


def run_pipeline():
    """主流程"""
    total_steps = 6 if FEATURE_ENGINEER and MODEL == "XGBoost" else 5

    # ============================================================
    # 环境初始化
    # ============================================================
    print_section("环境初始化", 1, total_steps)

    ensure_data()
    create_output_dirs()

    env_msg = "CLOUD" if IS_CLOUD else "LOCAL"
    print(f"运行环境: {env_msg}")

    # ============================================================
    # 步骤 1: 加载数据
    # ============================================================
    print_section("加载训练数据", 2, total_steps)
    train_df = load_train_csv()
    print(f"数据形状: {train_df.shape}")

    # ============================================================
    # 步骤 2: 数据预处理
    # ============================================================
    print_section("数据预处理", 3, total_steps)
    train_non_overlap = create_non_overlap_data(train_df)
    print(f"非重叠数据形状: {train_non_overlap.shape}")

    generate_spectrograms_npy()

    # ============================================================
    # 步骤 3 & 4: 特征工程（可选）
    # ============================================================
    if FEATURE_ENGINEER:
        if FEATURE_ENGINEER:
            print_section("特征工程", 4, total_steps)
            final_df = run_feature_engineering(train_non_overlap)
            if final_df is None:
                print("⚠️ 特征工程已跳过，使用已有特征文件")

    # ============================================================
    # 步骤 5: 模型训练
    # ============================================================
    if MODEL == "XGBoost":
        print_section("XGBoost 模型训练", 5, total_steps)
        df = load_features()
        X, y_encoded, groups, feature_cols, le = prepare_data(df, train_df)
        model, cv_results = train_xgboost_with_group_cv(
            X, y_encoded, groups, feature_cols
        )

    # ============================================================
    # 完成
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  ✅ 项目全部运行完成！")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_pipeline()
