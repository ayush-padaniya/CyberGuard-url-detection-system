import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from src.logger import logging
from config.config_mlflow import setup_mlflow

# ==============================================
#              Constants
# ==============================================
RUN_ID_PATH  = "artifacts/run_id.json"
METRICS_PATH = "artifacts/metrics.json"
MODEL_NAME   = "CyberGuard-XGBoost"
F1_THRESHOLD = 0.80


# ==============================================
#        Load Run ID
# ==============================================
def load_run_id() -> str:
    try:
        if not os.path.exists(RUN_ID_PATH):
            raise FileNotFoundError(f"run_id.json not found at {RUN_ID_PATH}")

        with open(RUN_ID_PATH, "r") as f:
            data = json.load(f)

        run_id = data["run_id"]
        logging.info(f"Run ID loaded: {run_id}")
        return run_id

    except Exception as e:
        logging.error(f"Error loading run_id: {e}")
        raise


# ==============================================
#        Load Metrics
# ==============================================
def load_metrics() -> dict:
    try:
        if not os.path.exists(METRICS_PATH):
            raise FileNotFoundError(f"metrics.json not found at {METRICS_PATH}")

        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        logging.info(f"Metrics loaded: {metrics}")
        return metrics

    except Exception as e:
        logging.error(f"Error loading metrics: {e}")
        raise


# ==============================================
#        Register Model
# ==============================================
def register_model(run_id: str) -> str:
    try:
        model_uri = f"runs:/{run_id}/model"

        # ── Register model → auto creates new version ──
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        logging.info(f"Model registered: {MODEL_NAME} v{model_version.version}")
        return model_version.version

    except Exception as e:
        logging.error(f"Error registering model: {e}")
        raise


# ==============================================
#        Promote to Production
# ==============================================
def promote_to_production(version: str):
    try:
        client = MlflowClient()

        # ── Set alias as Production ──
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Production",
            version=version
        )

        logging.info(f"Model v{version} promoted to Production alias")

    except Exception as e:
        logging.error(f"Error promoting model: {e}")
        raise


# ==============================================
#              Main
# ==============================================
def main():
    try:
        logging.info("=" * 60)
        logging.info("Starting Model Registry")
        logging.info("=" * 60)

        # Setup MLflow
        setup_mlflow("ayush-padaniya", "CyberGuard-url-detection-system")

        # Step 1 — Check metrics.json exists
        if not os.path.exists(METRICS_PATH):
            logging.error("metrics.json not found — model did not pass threshold")
            logging.error("Run Model_Evaluation.py first")
            return

        # Step 2 — Load metrics
        metrics = load_metrics()

        # Step 3 — Check threshold
        if metrics.get("f1_macro", 0) < F1_THRESHOLD:
            logging.warning(f"F1 score {metrics['f1_macro']} below threshold {F1_THRESHOLD}")
            logging.warning("Model will NOT be registered")
            return

        logging.info(f"F1 score {metrics['f1_macro']} passed threshold {F1_THRESHOLD}")

        # Step 4 — Load run_id
        run_id = load_run_id()

        # Step 5 — Register model (auto new version)
        version = register_model(run_id)

        # Step 6 — Promote to Production
        promote_to_production(version)

        logging.info("=" * 60)
        logging.info(f"Model v{version} registered and promoted to Production")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Model Registry Failed: {e}")
        raise


if __name__ == "__main__":
    main()