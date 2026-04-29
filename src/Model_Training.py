import os
import json
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier

from src.logger import logging
from config.config_mlflow import setup_mlflow
# ==============================================
#              Constants
# ==============================================
TRAIN_DATA_PATH = "data/preprocessed/train.csv"
ARTIFACTS_PATH  = "artifacts/run_id.json"


# ==============================================
#              Load Data
# ==============================================
def load_data():
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
        logging.info(f"Train data loaded: {df.shape}")

        X = df.drop(columns=["label"])
        y = df["label"]

        return X, y

    except Exception as e:
        logging.error(f"Error loading train data: {e}")
        raise


# ==============================================
#              Train Model
# ==============================================
def train_model(X, y):
    try:
        logging.info("Initializing XGBoost model with tuned parameters...")

        model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=1.0,
            min_child_weight=1,
            scale_pos_weight=3,
            tree_method="hist",
            device="cuda",   # 🔥 GPU enabled
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )

        model.fit(X, y)

        logging.info("Model training completed")

        return model

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise


# ==============================================
#          Save Run ID Locally
# ==============================================
def save_run_id(run_id: str):
    try:
        os.makedirs("artifacts", exist_ok=True)

        with open(ARTIFACTS_PATH, "w") as f:
            json.dump({"run_id": run_id}, f)

        logging.info(f"Run ID saved locally at {ARTIFACTS_PATH}")

    except Exception as e:
        logging.error(f"Error saving run_id: {e}")
        raise


# ==============================================
#              Main Function
# ==============================================
def main():
    try:
        logging.info("=" * 60)
        logging.info("🚀 Starting Model Training")
        logging.info("=" * 60)

        # 🔥 Setup MLflow (DagsHub)
        setup_mlflow("ayush-padaniya", "CyberGuard-url-detection-system")
        logging.info("MLflow setup completed")



        # ✅ Set experiment name
        mlflow.set_experiment("CyberGuard-XGBoost")
        logging.info("MLflow experiment set to 'CyberGuard-XGBoost'")


        # Load data
        X, y = load_data()
        logging.info("Data loaded successfully")


        # Start MLflow run
        with mlflow.start_run() as run:

            run_id = run.info.run_id
            logging.info(f"MLflow Run ID: {run_id}")

            # Train model
            model = train_model(X, y)
            logging.info("Model trained successfully")


            # Log parameters
            mlflow.log_params({
                "model": "XGBoost",
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 1.0,
                "min_child_weight": 1,
                "scale_pos_weight": 3,
                "tree_method": "hist",
                "device": "cuda"
            })
            logging.info("Parameters logged to MLflow")



            # Log model ONLY in MLflow
            mlflow.xgboost.log_model(model, "model")
            logging.info("Model logged to MLflow")



            # Save run_id locally
            save_run_id(run_id)
            logging.info("Run ID saved locally")



        logging.info("=" * 60)
        logging.info("✅ Model Training Completed")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"❌ Training Failed: {e}")
        raise


if __name__ == "__main__":
    main()