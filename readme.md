### 文件目录格式

```
+-- cache # 缓存位置
│   \-- specs.npy # 用于加快特征工程
+-- Data # 数据集位置
│   \-- hms-harmful-brain-activity-classification
+-- outputs # 输出位置
│   \-- train_with_features.pkl
+-- src # 源代码
│   +-- __pycache__
│   +-- __init__.py
│   +-- data_loader.py # 数据加载
│   +-- feature_engineering.py # 特征工程
│   +-- generate_npy.py # 生成 `npy` 文件
│   +-- preprocess.py # 数据预处理
│   \-- train_xgboost.py # 训练 XGBoost 模型
+-- .gitignore
+-- config.py # 环境变量
+-- download_data.py # 云端环境下载数据
+-- main.py # 主函数
+-- readme.md # 项目说明
+-- requirements.txt # 需求
\-- tmp.py # 测试文件
```

### 关于数据
+ `train.csv` 文件主要起到索引作用
+ `train_eegs/` 文件夹内的parquet文件为电极所记录的原始脑电波数据(可理解为音频)
+ `train_spectrograms/` 文件夹内的parquet文件为经过傅里叶变换得来的频谱图数据(可理解为分轨),  值为某个电极的某个频率在某个时间戳下的功率(可理解为音量)
  + 共有四百个频率(四百列),  每个文件的列名都相同
  + time 列为时间帧,  一帧对应两秒的时间窗口.
  + 傅里叶变换将时间窗口内的脑电波转化为各个频率的分布.

### 执行流程
+ 数据下载(如果是云端执行)并解压
+ 数据加载
+ 数据预处理
+ 生成 `*.npy` 文件,  加快后续速度
+ 特征工程
+ 模型训练

### 疑问
+ 竞赛方如何处理的时间窗口的边界效应

### 小补充
+ 文件I/O默认单线程,  运行非常慢,  后续加上了多线程明显加快了速度. 
+ 云端执行代码可能 `IS_CLOUD` 这个环境变量不会被设置成 `True` (我用的 Kaggle 免费 GPU),  需要在之前的单元格内手动设置该环境变量
+ 在云端运行整个流程时第一遍大概率出现内存不足的问题,  最简单的处理方式是再次运行单元格,  因为已经有避免重复的代码逻辑,  会跳过缓存文件生成和特征工程的步骤.
+ 对于索引文件指向真实数据的项目,  最简单可靠的处理办法是忍受第一次的耗时(缓存优化+特征工程),  在训练前对 merge 操作后的数据切片.  其他方法可能会带来索引文件和真实数据不能对齐的问题.  