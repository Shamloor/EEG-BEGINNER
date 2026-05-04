import pandas as pd
import os
from glob import glob
from config import DATA_PATH

def load_train_csv():
    """加载训练标签 csv"""
    path = os.path.join(DATA_PATH, 'train.csv')
    df = pd.read_csv(path)
    return df

def get_eeg_file_list():
    """获取所有 EEG parquet 文件"""
    path = os.path.join(DATA_PATH, 'train_eegs', '*.parquet')
    return glob(path)