# train_xgboost.py
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import warnings
import gc
from config import OUTPUT_PKL_PATH, XGB_MODEL_PATH

warnings.filterwarnings("ignore")


def load_features():
    """加载特征工程生成的pkl文件"""
    print("正在加载特征数据...")
    df = pd.read_pickle(OUTPUT_PKL_PATH)
    print(f"特征数据形状: {df.shape}")
    return df


def prepare_data(df, labels_df):
    """
    准备XGBoost训练所需的数据，通过与标签文件映射获取目标变量

    Args:
        df: 特征数据（从特征工程生成，并包含 spec_id, patient_id 等列）
        labels_df: 标签数据（从 train.csv 读取，包含 spectrogram_id, expert_consensus 等列）

    Returns:
        X: 特征矩阵
        y: 标签数组
        groups: 分组信息（patient_id）
        feature_cols: 特征列名列表
        le: 标签编码器
    """
    # 识别特征列 - 排除非特征列
    exclude_cols = ["spec_id", "min_time", "max_time", "patient_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 合并标签数据
    # 注意：特征数据中的 'spec_id' 对应标签数据中的 'spectrogram_id'
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

    # 获取标签（expert_consensus）
    y = merged_df["expert_consensus"].values

    # 提取分组信息（用于 GroupKFold，防止病人级别的数据泄露）
    groups = (
        merged_df["patient_id"].values if "patient_id" in merged_df.columns else None
    )

    # 编码多分类标签（expert_consensus 是字符串类型）
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"标签分布:")
    for i, label in enumerate(le.classes_):
        count = (y_encoded == i).sum()
        print(f"  {label}: {count} ({count/len(y_encoded)*100:.1f}%)")

    if groups is not None:
        print(f"共有 {len(np.unique(groups))} 个不同的 patient_id")

    # 删除不再需要的 merged_df，释放内存
    del merged_df
    gc.collect()

    return X, y_encoded, groups, feature_cols, le


def train_xgboost_with_group_cv(X, y, groups, feature_cols):
    """
    使用 GroupKFold 进行交叉验证训练 XGBoost 模型

    Args:
        X: 特征矩阵
        y: 标签数组
        groups: 分组信息 (patient_id)
        feature_cols: 特征列名列表 (仅用于记录)

    Returns:
        model: 在全量数据上训练好的模型
        cv_results: 交叉验证的详细结果
    """
    from sklearn.metrics import classification_report
    import xgboost as xgb

    n_classes = len(np.unique(y))
    print(f"\n使用 GroupKFold 进行 {n_classes} 分类训练")

    # XGBoost 参数配置 (可根据验证结果调整)
    params = {
        # 损失函数, 也就是模型优化的目标, 采用多分类还是二分类
        "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
        # 评估指标, 衡量验证集上的性能质量
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        # 多分类时制定类别总数
        "num_class": n_classes if n_classes > 2 else None,
        # 最大深度
        "max_depth": 8,
        # 学习率
        "learning_rate": 0.05,
        # 增益函数中的正则项, 越大则越难分裂, 泛化能力越强
        "gamma": 0.1,
        # 叶子节点的L1惩罚
        "reg_alpha": 0.1,
        # 叶子节点的L2惩罚
        "reg_lambda": 1,
        # 叶子节点至少需要的样本权重和, 值越大, 模型越保守, 防止学习到局部异常的噪声
        "min_child_weight": 3,
        # 每棵树使用的样本比例
        "subsample": 0.8,
        # 每棵树使用的特征比例
        "colsample_bytree": 0.8,
        "seed": 42,
        # GPU 加速配置
        "tree_method": "hist",  # 使用 GPU 直方图算法
        "device": "cuda",  # 指定使用 GPU
        "predictor": "gpu_predictor",  # 预测时也使用 GPU（可选，加速预测）
        # CPU 参数可以降低或删除，因为用 GPU 时 CPU 不是瓶颈
        "n_jobs": 1,  # GPU 模式下建议设为 1，避免资源竞争
    }

    # 初始化分组交叉验证 (例如 5 折)
    n_splits = 5
    group_kfold = GroupKFold(n_splits=n_splits)

    print(f"开始 {n_splits} 折 GroupKFold 交叉验证...")

    # 用于记录每折的验证分数
    cv_scores = []
    fold = 1

    # 存储每折的预测概率，用于整体评估
    # 折外预测
    all_oof_predictions = (
        np.zeros((len(y), n_classes)) if n_classes > 2 else np.zeros(len(y))
    )

    # 直接使用 y，不需要 copy（y 不会被修改）
    all_true_labels = y

    # 将 np.unique(y) 移到循环外，避免重复计算
    all_classes = np.unique(y)

    # 手动进行交叉验证循环，以便观察每折结果
    for train_idx, val_idx in group_kfold.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 预测验证集
        y_val_pred_proba = model.predict_proba(X_val)

        # 存储袋外预测 (OOF)
        all_oof_predictions[val_idx] = y_val_pred_proba

        # 计算对数损失 (多分类常用)
        loss = log_loss(y_val, y_val_pred_proba, labels=all_classes)
        cv_scores.append(loss)

        print(f"  Fold {fold}/{n_splits} - Log Loss: {loss:.4f}")
        fold += 1

        # 释放当前折的模型，避免 GPU 显存累积
        del model
        gc.collect()

    # 输出整体交叉验证性能
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"\nGroupKFold 交叉验证结果 (5折):")
    print(f"  平均 Log Loss: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")

    # 计算整体袋外预测的分类报告
    oof_predictions = (
        np.argmax(all_oof_predictions, axis=1)
        if n_classes > 2
        else (all_oof_predictions > 0.5).astype(int)
    )
    print("\n=== 整体袋外预测 (Out-of-Fold) 分类报告 ===")
    print(classification_report(all_true_labels, oof_predictions))

    # 最终在全量数据上训练一个最终模型
    print("\n使用全部数据训练最终模型...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    # 返回最终模型和交叉验证结果
    return final_model, {
        "cv_scores": cv_scores,
        "mean_score": mean_cv_score,
        "std_score": std_cv_score,
    }


def save_model(model):
    """保存训练好的模型"""
    joblib.dump(model, XGB_MODEL_PATH)
    print(f"模型已保存到: {XGB_MODEL_PATH}")
