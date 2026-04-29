import pandas as pd
from src.logger import logging


# your check_leakage.py reads processed files
TRAIN_PATH = "data/processed/train.csv"   # ← Feature Engineering output
TEST_PATH  = "data/processed/test.csv"    # ← Feature Engineering output


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def check_overlap(train_df, test_df):
    logging.info("Checking train-test overlap...")

    train_hashes = pd.util.hash_pandas_object(train_df, index=False)
    test_hashes = pd.util.hash_pandas_object(test_df, index=False)

    overlap = set(train_hashes).intersection(set(test_hashes))

    if len(overlap) > 0:
        logging.error(f"❌ LEAKAGE FOUND: {len(overlap)} duplicate rows")
    else:
        logging.info("✅ No overlap between train and test")


def check_label_presence(df):
    if "label" in df.columns:
        logging.info("Label column exists (expected)")
    else:
        logging.warning("⚠️ Label column missing")


def check_constant_columns(df, name):
    logging.info(f"Checking constant columns in {name}...")

    for col in df.columns:
        if df[col].nunique() == 1:
            logging.warning(f"⚠️ Constant column detected: {col}")


def main():
    logging.info("=" * 60)
    logging.info("🚨 DATA LEAKAGE CHECK STARTED")
    logging.info("=" * 60)

    train_df, test_df = load_data()

    logging.info(f"Train shape: {train_df.shape}")
    logging.info(f"Test shape: {test_df.shape}")

    check_overlap(train_df, test_df)

    check_label_presence(train_df)
    check_label_presence(test_df)

    check_constant_columns(train_df, "TRAIN")
    check_constant_columns(test_df, "TEST")

    logging.info("=" * 60)
    logging.info("✅ CHECK COMPLETED")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()