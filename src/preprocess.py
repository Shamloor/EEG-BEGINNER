def create_non_overlap_data(train_df):
    """按 eeg_id 分组，生成非重叠样本"""
    grouped = train_df.groupby("eeg_id")

    agg_dict = {"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"}

    df = grouped[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(agg_dict)
    df.columns = ["spec_id", "min_time"]
    df["max_time"] = grouped["spectrogram_label_offset_seconds"].max()
    df["patient_id"] = grouped["patient_id"].first()

    return df
