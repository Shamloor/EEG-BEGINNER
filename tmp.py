# from src.data_loader import load_train_csv
# from src.preprocess import create_non_overlap_data
# from src.generate_h5 import generate_spectrograms_h5
# from src.feature_engineering import run_feature_engineering, load_spectrograms
# from src.train_xgboost import load_features, prepare_data, train_xgboost_with_group_cv
# from config import FEATURE_ENGINEER, MODEL, IS_CLOUD, OUTPUT_PKL_PATH, CACHE_SPECS_H5
# import pandas as pd
# import h5py

# if __name__ == "__main__":
#     generate_spectrograms_h5()

    # train_df = load_train_csv()
    # train_non_overlap = create_non_overlap_data(train_df)

    # run_feature_engineering(train_non_overlap)

    # print(str(train_non_overlap.iloc[0].spec_id))

    # spectrograms = load_spectrograms()
    # h5_keys = list(spectrograms.keys())[:5]
    # print(type(h5_keys[0]))

from memory_profiler import profile

@profile
def your_function():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    print("adfsdfdsf")
    return a

if __name__ == '__main__':
    your_function()