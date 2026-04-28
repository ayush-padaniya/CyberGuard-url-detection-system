import os
import pandas as pd
from src.logger import logging

# ==============================================
#              Constants
# ==============================================
TRAIN_DATA_PATH = 'data/raw/train.csv'
TEST_DATA_PATH  = 'data/raw/test.csv'

PROCESSED_TRAIN_PATH = 'data/processed/train.csv'
PROCESSED_TEST_PATH  = 'data/processed/test.csv'

COLUMNS_TO_DROP = ['url', 'domain', 'scan_date', 'type']


# ==============================================
#     RAW URL → FEATURE CONVERSION (NEW ADD)
# ==============================================
def extract_features_from_url(url: str) -> dict:
    """Convert raw URL into numeric features"""

    try:
        features = {}

        # Basic features
        features['url_len'] = len(url)
        features['digits'] = sum(c.isdigit() for c in url)
        features['letters'] = sum(c.isalpha() for c in url)

        # Special characters
        features['@'] = url.count('@')
        features['?'] = url.count('?')
        features['-'] = url.count('-')
        features['='] = url.count('=')
        features['.'] = url.count('.')
        features['#'] = url.count('#')
        features['%'] = url.count('%')
        features['+'] = url.count('+')
        features['$'] = url.count('$')
        features['!'] = url.count('!')
        features['*'] = url.count('*')
        features[','] = url.count(',')
        features['//'] = url.count('//')

        # Security signals
        features['https'] = 1 if "https" in url else 0
        features['having_ip_address'] = 1 if any(char.isdigit() for char in url.split('/')[0]) else 0
        features['abnormal_url'] = 1 if "@" in url else 0

        return features

    except Exception as e:
        logging.error(f"Error extracting features from URL: {e}")
        raise


# ==============================================
#        Feature Engineering Function
# ==============================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and column selection."""

    try:
        logging.info("Starting Feature Engineering...")

        # ── Handle RAW URL case (important for real-world input) ──
        if 'url' in df.columns and 'digits' not in df.columns:
            logging.info("Extracting features from raw URLs...")
            feature_df = df['url'].apply(extract_features_from_url).apply(pd.Series)
            df = pd.concat([df, feature_df], axis=1)

        # ── Drop unwanted columns ──
        df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
        logging.info(f"Dropped columns: {COLUMNS_TO_DROP}")

        # =========================================================
        #            Add Meaningful Features
        # =========================================================

        # 1️⃣ Digit to letter ratio
        if 'digits' in df.columns and 'letters' in df.columns:
            df['digit_letter_ratio'] = df['digits'] / (df['letters'] + 1)
            logging.info("Created: digit_letter_ratio")

        # 2️⃣ Special character density
        special_cols = ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',', '//']
        existing_special_cols = [col for col in special_cols if col in df.columns]

        if len(existing_special_cols) > 0 and 'url_len' in df.columns:
            df['special_char_ratio'] = df[existing_special_cols].sum(axis=1) / (df['url_len'] + 1)
            logging.info("Created: special_char_ratio")

        # 3️⃣ URL complexity score
        if 'url_len' in df.columns and 'digits' in df.columns:
            df['url_complexity'] = df['url_len'] * (df['digits'] + 1)
            logging.info("Created: url_complexity")

        logging.info(f"Final dataset shape after FE: {df.shape}")

        return df

    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        raise


# ==============================================
#              Save Data
# ==============================================
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save processed train and test data."""

    try:
        os.makedirs('data/processed', exist_ok=True)

        train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
        test_df.to_csv(PROCESSED_TEST_PATH, index=False)

        logging.info(f"Processed train saved to {PROCESSED_TRAIN_PATH}")
        logging.info(f"Processed test saved to {PROCESSED_TEST_PATH}")

    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise


# ==============================================
#              Main Function
# ==============================================
def main():
    try:
        logging.info("=" * 60)
        logging.info("🚀 Starting Feature Engineering")
        logging.info("=" * 60)

        # Load data
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df  = pd.read_csv(TEST_DATA_PATH)

        logging.info("Data loaded successfully")

        # Apply feature engineering
        train_df = feature_engineering(train_df)
        test_df  = feature_engineering(test_df)

        # Save processed data
        save_data(train_df, test_df)

        logging.info("=" * 60)
        logging.info("✅ Feature Engineering Completed")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"❌ Feature Engineering Failed: {e}")
        raise

if __name__ == "__main__":
    main()