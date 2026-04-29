import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.logger import logging
from src.connections.s3_connections import download_file_from_s3
import yaml

# ── Load params ──
with open('params.yaml', 'r',  encoding="utf-8") as f:
    params = yaml.safe_load(f)

# ==============================================
#                Constants
# ==============================================
RAW_DATA_PATH     = 'data/raw/merged.csv'
TRAIN_DATA_PATH   = 'data/raw/train.csv'
TEST_DATA_PATH    = 'data/raw/test.csv'
S3_KEY        = params['data_ingestion']['s3_key']        # ← from yaml
TEST_SIZE     = params['data_ingestion']['test_size']     # ← from yaml
RANDOM_STATE  = params['data_ingestion']['random_state']  # ← from yaml

# ==============================================
#        Download data from S3 bucket
# ==============================================

def download_data_from_s3() -> None:
    try:
        if os.path.exists(RAW_DATA_PATH):
            logging.info(f"Data already exists at {RAW_DATA_PATH} — skipping download")
            return

        bucket_name = os.getenv('S3_BUCKET_NAME')
        if not bucket_name:
            raise EnvironmentError("S3_BUCKET_NAME environment variable is not set")

        os.makedirs('data/raw', exist_ok=True)

        logging.info(f"Downloading data from S3 bucket: {bucket_name}")

        download_file_from_s3(
            bucket_name=bucket_name,
            s3_key=S3_KEY,
            local_path=RAW_DATA_PATH
        )

        logging.info(f"Data downloaded successfully to {RAW_DATA_PATH}")

    except Exception as e:
        logging.error(f"Error downloading data from S3: {e}")
        raise


# ==============================================
#            Load dataset
# ==============================================

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        logging.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


# ==============================================
#        Leakage-safe Split Function
# ==============================================

def split_data(df: pd.DataFrame) -> tuple:
    try:
        logging.info("Starting leakage-safe split using URL grouping...")

        splitter = GroupShuffleSplit(
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        train_idx, test_idx = next(splitter.split(df, groups=df["url"]))

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # ======================================
        # FINAL LEAKAGE CHECK (VERY IMPORTANT)
        # ======================================
        overlap = set(train_df["url"]).intersection(set(test_df["url"]))

        logging.info(f"Leakage check (URL overlap): {len(overlap)}")

        if len(overlap) > 0:
            raise ValueError("❌ DATA LEAKAGE DETECTED — STOPPING PIPELINE")

        logging.info("Leakage-free split confirmed ✅")

        return train_df, test_df

    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise


# ==============================================
#              Save Data
# ==============================================

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)

        logging.info(f"Train data saved to {TRAIN_DATA_PATH}")
        logging.info(f"Test data saved to {TEST_DATA_PATH}")

    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


# ==============================================
#              Main Pipeline
# ==============================================

def main():
    try:
        logging.info("=" * 60)
        logging.info("Starting Data Ingestion Pipeline")
        logging.info("=" * 60)

        # Skip if already exists
        if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH):
            logging.info("Train/Test already exist — skipping ingestion")
            return

        # Step 1: Download
        download_data_from_s3()

        # Step 2: Load
        df = load_data(RAW_DATA_PATH)

        # Step 3: REMOVE DUPLICATE URLS (IMPORTANT FIX)
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
        logging.info(f"After URL deduplication: {df.shape}")

        # Step 4: Split (LEAKAGE SAFE)
        train_df, test_df = split_data(df)

        # Step 5: Save
        save_data(train_df, test_df)

        logging.info("=" * 60)
        logging.info("Data Ingestion Completed Successfully ✅")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Data Ingestion Failed: {e}")
        raise


if __name__ == '__main__':
    main()