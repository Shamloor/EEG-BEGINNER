import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from src.base_trainer import group_kfold_cv_base


def train_catboost_with_group_cv(X, y, groups):
    """使用 CatBoost 训练"""
    n_classes = len(np.unique(y))

    # CatBoost 参数配置
    params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 8,
        "loss_function": "MultiClass" if n_classes > 2 else "Logloss",
        "eval_metric": "MultiClass" if n_classes > 2 else "Logloss",
        # 删除 custom_metric 这一行
        "random_seed": 42,
        "verbose": 100,
        "early_stopping_rounds": 50,
        "task_type": "GPU",
        "devices": "0",
        "l2_leaf_reg": 3,
        "border_count": 128,
        "auto_class_weights": "Balanced",
    }

    # 交叉验证
    cv_scores, all_oof_predictions, all_true_labels = group_kfold_cv_base(
        X, y, groups, CatBoostClassifier, params, n_splits=5
    )

    # 输出结果
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"\nGroupKFold 交叉验证结果:")
    print(f"  平均 Log Loss: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")

    # 计算整体袋外预测
    n_classes = len(np.unique(y))
    oof_predictions = (
        np.argmax(all_oof_predictions, axis=1)
        if n_classes > 2
        else (all_oof_predictions > 0.5).astype(int)
    )
    print("\n=== 整体袋外预测 (Out-of-Fold) 分类报告 ===")
    print(classification_report(all_true_labels, oof_predictions))

    # 训练最终模型
    print("\n使用全部数据训练最终模型...")
    final_model = CatBoostClassifier(**params)
    final_model.fit(X, y, verbose=100)

    return final_model, {
        "cv_scores": cv_scores,
        "mean_score": mean_cv_score,
        "std_score": std_cv_score,
    }
