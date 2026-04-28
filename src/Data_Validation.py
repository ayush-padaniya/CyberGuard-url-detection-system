import pandas as pd
import great_expectations as ge
from src.logger import logging

# ══════════════════════════════════════════════
#              Constants
# ══════════════════════════════════════════════
REQUIRED_COLUMNS = [
    'url_len', 'label', 'https',
    'having_ip_address', 'abnormal_url',
    'digits', 'letters'
]

def validate_data(file_path: str) -> bool:
    """Validate dataset using Great Expectations."""
    try:
        logging.info("=" * 60)
        logging.info("Starting Data Validation")
        logging.info("=" * 60)

        # ── Load data as GE dataframe ──
        df = ge.read_csv(file_path)
        logging.info(f"Data loaded for validation: {file_path}")

        # ── Expectation 1: Row count ──
        result = df.expect_table_row_count_to_be_between(
            min_value=1000
        )
        logging.info(f"Row count check: {result['success']}")

        # ── Expectation 2: Required columns exist ──
        for col in REQUIRED_COLUMNS:
            result = df.expect_column_to_exist(col)
            logging.info(f"Column '{col}' exists: {result['success']}")

        # ── Expectation 3: No nulls in label ──
        result = df.expect_column_values_to_not_be_null('label')
        logging.info(f"No nulls in label: {result['success']}")

        # ── Expectation 4: Label values valid ──
        result = df.expect_column_values_to_be_in_set(
            'label', [0, 1, 2, 3]
        )
        logging.info(f"Label values valid: {result['success']}")

        # ── Expectation 5: url_len positive ──
        result = df.expect_column_values_to_be_between(
            'url_len', min_value=0
        )
        logging.info(f"url_len positive: {result['success']}")

        # ── Expectation 6: https binary ──
        result = df.expect_column_values_to_be_in_set(
            'https', [0, 1]
        )
        logging.info(f"https binary: {result['success']}")

        # ── Expectation 7: No nulls in features ──
        for col in REQUIRED_COLUMNS:
            result = df.expect_column_values_to_not_be_null(col)
            logging.info(f"No nulls in '{col}': {result['success']}")

        # ── Expectation 8: url_len reasonable range ──
        result = df.expect_column_values_to_be_between(
            'url_len', min_value=1, max_value=10000
        )
        logging.info(f"url_len range valid: {result['success']}")

        logging.info("=" * 60)
        logging.info("✅ Data Validation Passed!")
        logging.info("=" * 60)
        return True

    except Exception as e:
        logging.error(f"❌ Data Validation Failed: {e}")
        raise


# ══════════════════════════════════════════════
#              Runner
# ══════════════════════════════════════════════
if __name__ == "__main__":
    logging.info("🔍 Validating TRAIN dataset")
    validate_data("data/raw/train.csv")

    logging.info("🔍 Validating TEST dataset")
    validate_data("data/raw/test.csv")