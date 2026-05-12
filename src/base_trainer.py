"""训练相关的基础功能和工具函数"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import warnings
import gc
from config import OUTPUT_PKL_PATH, TEST_MODE

warnings.filterwarnings("ignore")


def load_features():
    """加载特征工程生成的pkl文件"""
    print("正在加载特征数据...")
    df = pd.read_pickle(OUTPUT_PKL_PATH)
    print(f"特征数据形状: {df.shape}")
    return df


def prepare_data(df, labels_df):
    """
    准备训练所需的数据，通过与标签文件映射获取目标变量

    Args:
        df: 特征数据（包含 spec_id, patient_id 等列）
        labels_df: 标签数据（包含 spectrogram_id, expert_consensus 等列）

    Returns:
        X: 特征矩阵
        y_encoded: 编码后的标签数组
        groups: 分组信息（patient_id）
        feature_cols: 特征列名列表
        le: 标签编码器
    """
    # 识别特征列 - 排除非特征列
    exclude_cols = ["spec_id", "min_time", "max_time", "patient_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 合并标签数据
    merged_df = df.merge(
        labels_df[["spectrogram_id", "expert_consensus"]],
        left_on="spec_id",
        right_on="spectrogram_id",
        how="inner",
    )

    print(f"特征数据原始条数: {len(df)}")
    print(f"合并后条数: {len(merged_df)}")

    if len(merged_df) == 0:
        raise ValueError(
            "特征数据和标签数据无法匹配，请检查 spec_id 和 spectrogram_id 的对应关系"
        )

    # 提取特征矩阵
    X = merged_df[feature_cols].values

    # 获取标签
    y = merged_df["expert_consensus"].values

    # 提取分组信息
    groups = (
        merged_df["patient_id"].values if "patient_id" in merged_df.columns else None
    )

    # 编码多分类标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"标签分布:")
    for i, label in enumerate(le.classes_):
        count = (y_encoded == i).sum()
        print(f"  {label}: {count} ({count/len(y_encoded)*100:.1f}%)")

    if groups is not None:
        print(f"共有 {len(np.unique(groups))} 个不同的 patient_id")

    # ========== 切片逻辑 ==========
    if TEST_MODE:
        SAMPLE_SIZE = 1000
        np.random.seed(42)
        indices = np.random.choice(len(X), min(SAMPLE_SIZE, len(X)), replace=False)
        X = X[indices]
        y_encoded = y_encoded[indices]
        if groups is not None:
            groups = groups[indices]
        print(f"⚠️ 测试模式：仅使用 {len(X)} 个样本进行训练")

    # 释放内存
    del merged_df
    gc.collect()

    return X, y_encoded, groups, feature_cols, le


def group_kfold_cv_base(X, y, groups, model_class, model_params, n_splits=5):
    """
    通用的 GroupKFold 交叉验证框架

    Args:
        X: 特征矩阵
        y: 标签数组
        groups: 分组信息
        model_class: 模型类（如 xgb.XGBClassifier 或 CatBoostClassifier）
        model_params: 模型参数字典
        n_splits: 交叉验证折数

    Returns:
        cv_scores: 每折的分数列表
        all_oof_predictions: 袋外预测概率
        all_true_labels: 真实标签
    """
    n_classes = len(np.unique(y))
    group_kfold = GroupKFold(n_splits=n_splits)

    cv_scores = []
    all_oof_predictions = (
        np.zeros((len(y), n_classes)) if n_classes > 2 else np.zeros(len(y))
    )
    all_true_labels = y
    all_classes = np.unique(y)

    print(f"开始 {n_splits} 折 GroupKFold 交叉验证...")

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # 预测验证集
        y_val_pred_proba = model.predict_proba(X_val)

        # 存储袋外预测
        all_oof_predictions[val_idx] = y_val_pred_proba

        # 计算对数损失
        loss = log_loss(y_val, y_val_pred_proba, labels=all_classes)
        cv_scores.append(loss)

        print(f"  Fold {fold}/{n_splits} - Log Loss: {loss:.4f}")

        # 释放模型内存
        del model
        gc.collect()

    return cv_scores, all_oof_predictions, all_true_labels


def save_model(model, model_path):
    """保存训练好的模型"""
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")
