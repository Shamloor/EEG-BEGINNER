from src.data_loader import load_train_csv
from src.preprocess import create_non_overlap_data
from src.generate_h5 import generate_spectrograms_h5
from src.feature_engineering import run_feature_engineering
from src.base_trainer import load_features, prepare_data, save_model
from src.train_xgboost import train_xgboost_with_group_cv
from src.train_catboost import train_catboost_with_group_cv
from config import FEATURE_ENGINEER, MODEL, IS_CLOUD
import tracemalloc
import gc


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


def start_memory_monitor():
    """启动内存监测"""
    tracemalloc.start()


def print_memory_report():
    """打印内存报告"""
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("\n" + "=" * 60)
    print("内存占用 Top 10:")
    for stat in top_stats[:10]:
        print(
            f"  {stat.size / 1024 / 1024:.2f} MB - {stat.traceback.format()[-1].strip()}"
        )
    print("=" * 60)

    tracemalloc.stop()


def run_pipeline():
    """主流程"""
    start_memory_monitor()
    total_steps = 6 if FEATURE_ENGINEER and MODEL == "XGBoost" else 5

    # ============================================================
    # 环境初始化
    # ============================================================
    print_section("环境初始化", 1, total_steps)

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
    # 索引变量
    train_non_overlap = create_non_overlap_data(train_df)
    print(f"非重叠数据形状: {train_non_overlap.shape}")

    generate_spectrograms_h5()

    # ============================================================
    # 步骤 3 & 4: 特征工程（可选）
    # ============================================================
    if FEATURE_ENGINEER:
        print_section("特征工程", 4, total_steps)
        final_df = run_feature_engineering(train_non_overlap)
        if final_df is None:
            print("⚠️ 特征工程已跳过，使用已有特征文件")
        else:
            del final_df
            gc.collect()

    del train_non_overlap
    gc.collect()

    # ============================================================
    # 步骤 5: 模型训练
    # ============================================================

    print_section("模型训练", 5, total_steps)
    df = load_features()
    X, y_encoded, groups, feature_cols, le = prepare_data(df, train_df)
    del df, train_df, feature_cols
    gc.collect()
    if MODEL == "XGBoost":
        model, cv_results = train_xgboost_with_group_cv(X, y_encoded, groups)
    elif MODEL == "CatBoost":
        model, cv_results = train_catboost_with_group_cv(X, y_encoded, groups)
    elif MODEL == "All":
        model_xgboost, cv_results_xgboost = train_xgboost_with_group_cv(
            X, y_encoded, groups
        )
        model_catboost, cv_results_catboost = train_catboost_with_group_cv(
            X, y_encoded, groups
        )

    # ============================================================
    # 完成
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  ✅ 项目全部运行完成！")
    print(f"{'=' * 60}\n")
    print_memory_report()


if __name__ == "__main__":
    run_pipeline()
