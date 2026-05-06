from src.data_loader import load_train_csv
from src.preprocess import create_non_overlap_data
from src.generate_h5 import generate_spectrograms_h5
from src.feature_engineering import run_feature_engineering
from src.train_xgboost import load_features, prepare_data, train_xgboost_with_group_cv
from config import FEATURE_ENGINEER, MODEL, IS_CLOUD

if __name__ == "__main__":
    # generate_spectrograms_h5()
    train_df = load_train_csv()
    train_non_overlap = create_non_overlap_data(train_df)
    run_feature_engineering(train_non_overlap)