import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from src.logger import logging

REFERENCE_DATA_PATH = "data/preprocessed/train.csv"
CURRENT_DATA_PATH   = "data/preprocessed/test.csv"
REPORT_DIR          = "monitoring/reports"
REPORT_PATH         = f"{REPORT_DIR}/evidently_report.html"


def run_evidently_report(predictions=None):
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)

        reference = pd.read_csv(REFERENCE_DATA_PATH)
        current   = pd.read_csv(CURRENT_DATA_PATH)

        # ── Add predictions to current data if available ──
        if predictions is not None:
            current["prediction"] = predictions

        column_mapping = ColumnMapping(
            target="label",
            prediction="prediction" if predictions is not None else None,
            numerical_features=[
                'url_len', 'digits', 'letters', 'digit_letter_ratio',
                'special_char_ratio', 'url_complexity',
                '@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ','
            ],
            categorical_features=[
                'https', 'having_ip_address', 'abnormal_url',
                'Shortining_Service', 'phish_suspicious_tld',
                'suspicious_extension', 'is_gov_edu'
            ]
        )

        # ── Build report based on predictions availability ──
        if predictions is not None:
            metrics = [
                DataDriftPreset(),       # feature drift detection
                TargetDriftPreset(),     # label/target drift detection
                ClassificationPreset(),  # model performance metrics
            ]
        else:
            metrics = [
                DataDriftPreset(),       # feature drift detection
                TargetDriftPreset(),     # label/target drift detection
            ]

        report = Report(metrics=metrics)

        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping
        )

        report.save_html(REPORT_PATH)
        logging.info(f"Evidently report saved to {REPORT_PATH}")

    except Exception as e:
        logging.error(f"Evidently report failed: {e}")
        raise


if __name__ == "__main__":
    run_evidently_report()