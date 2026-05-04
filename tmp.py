import xgboost as xgb
import numpy as np

X = np.random.rand(1000, 10)
y = np.random.randint(0, 3, 1000)

# 测试 GPU
model = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=10)
model.fit(X, y)
print("GPU 可用！")