import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.connections.s3_connections import download_file_from_s3

# ==============================================
#                Constants
# ==============================================
RAW_DATA_PATH     = 'data/raw/merged.csv'
TRAIN_DATA_PATH   = 'data/raw/train.csv'
TEST_DATA_PATH    = 'data/raw/test.csv'
S3_KEY            = 'data/Malicious-Url.csv'
TEST_SIZE         = 0.25
RANDOM_STATE      = 42


# ==============================================
#        Download data from s3 bucket
# ==============================================

def download_data_from_s3() -> None:
    """Download raw dataset from AWS S3 bucket only if not exists locally."""
    try:
        # ── Skip download if file already exists locally ──
        if os.path.exists(RAW_DATA_PATH):
            logging.info(f"Data already exists at {RAW_DATA_PATH} — skipping download ✅")
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
#            loads the dataset
# ==============================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from local CSV file."""
    try:
        df = pd.read_csv(file_path)  # ← change parquet to csv
        logging.info(f"Data loaded successfully from {file_path}")
        logging.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

# ==============================================
#       Splits Data into Train And Test 
# ==============================================

def split_data(df: pd.DataFrame) -> tuple:
    """Split dataset into train and test sets."""
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        logging.info(f"Data split completed")
        logging.info(f"Train size: {train_df.shape}")
        logging.info(f"Test size:  {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

# ==============================================
#       Saving Data into data/raw folder
# ==============================================

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train and test datasets to local CSV files."""
    try:
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        logging.info(f"Train data saved to {TRAIN_DATA_PATH}")
        logging.info(f"Test data saved to  {TEST_DATA_PATH}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


# ==============================================
#             Main Function
# ==============================================
def main():
    try:
        logging.info("=" * 60)
        logging.info("Starting Data Ingestion Pipeline")
        logging.info("=" * 60)

        # ── Skip everything if train and test already exist ──
        if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH):
            logging.info("Train and Test data already exist — skipping ingestion ✅")
            logging.info("=" * 60)
            return

        # Step 1 — Download from S3
        download_data_from_s3()

        # Step 2 — Load data
        df = load_data(RAW_DATA_PATH)

        # Step 3 — Split into train and test
        train_df, test_df = split_data(df)

        # Step 4 — Save splits locally
        save_data(train_df, test_df)

        logging.info("=" * 60)
        logging.info("Data Ingestion Completed Successfully ✅")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Data Ingestion Failed: t{e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()