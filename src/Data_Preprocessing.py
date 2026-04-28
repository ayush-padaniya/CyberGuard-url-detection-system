import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.logger import logging

# ==============================================
#              Constants
# ==============================================
PROCESSED_TRAIN_PATH = 'data/processed/train.csv'
PROCESSED_TEST_PATH  = 'data/processed/test.csv'

PREPROCESSED_TRAIN_PATH = 'data/preprocessed/train.csv'
PREPROCESSED_TEST_PATH  = 'data/preprocessed/test.csv'

PREPROCESSOR_PATH = 'artifacts/preprocessing.pkl'


# ==============================================
#        Preprocessing Function
# ==============================================
def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):

    try:
        logging.info("Starting Data Preprocessing...")

        # ── Separate target ──
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']

        X_test = test_df.drop(columns=['label'])
        y_test = test_df['label']

        # ── Initialize scaler ──
        scaler = StandardScaler()

        # ── Fit ONLY on train ──
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ── Convert back to DataFrame ──
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test  = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # ── Add label back ──
        X_train['label'] = y_train.values
        X_test['label']  = y_test.values

        logging.info("Scaling completed successfully")

        return X_train, X_test, scaler

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise


# ==============================================
#              Save Data & Scaler
# ==============================================
def save_data(train_df, test_df, scaler):

    try:
        os.makedirs('data/preprocessed', exist_ok=True)
        os.makedirs('artifacts', exist_ok=True)

        # Save datasets
        train_df.to_csv(PREPROCESSED_TRAIN_PATH, index=False)
        test_df.to_csv(PREPROCESSED_TEST_PATH, index=False)

        # Save scaler
        joblib.dump(scaler, PREPROCESSOR_PATH)

        logging.info(f"Preprocessed train saved to {PREPROCESSED_TRAIN_PATH}")
        logging.info(f"Preprocessed test saved to {PREPROCESSED_TEST_PATH}")
        logging.info(f"Scaler saved to {PREPROCESSOR_PATH}")

    except Exception as e:
        logging.error(f"Error saving preprocessing outputs: {e}")
        raise


# ==============================================
#              Main Function
# ==============================================
def main():
    try:
        logging.info("=" * 60)
        logging.info("🚀 Starting Data Preprocessing")
        logging.info("=" * 60)

        # Load processed data
        train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
        test_df  = pd.read_csv(PROCESSED_TEST_PATH)

        logging.info("Processed data loaded")

        # Apply preprocessing
        train_df, test_df, scaler = preprocess_data(train_df, test_df)

        # Save everything
        save_data(train_df, test_df, scaler)

        logging.info("=" * 60)
        logging.info("✅ Data Preprocessing Completed")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"❌ Preprocessing Failed: {e}")
        raise

if __name__ == "__main__":
    main()