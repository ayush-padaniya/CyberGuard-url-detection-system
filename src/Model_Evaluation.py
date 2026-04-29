import os
import json
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

from src.logger import logging
from config.config_mlflow import setup_mlflow
import yaml


# ── Load params ──
with open('params.yaml', 'r', encoding="utf-8") as f:
    params = yaml.safe_load(f)

# ==============================================
#              Constants
# ==============================================
TEST_DATA_PATH = "data/preprocessed/test.csv"
RUN_ID_PATH    = "artifacts/run_id.json"
METRICS_PATH   = "artifacts/metrics.json"
PLOTS_DIR      = "artifacts/plots"

F1_THRESHOLD   = params['model_evaluation']['f1_threshold']


# ==============================================
#        Load Run ID
# ==============================================
def load_run_id():
    with open(RUN_ID_PATH, "r") as f:
        data = json.load(f)
    return data["run_id"]


# ==============================================
#        Load Data
# ==============================================
def load_data():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


# ==============================================
#        Evaluate Model
# ==============================================
def evaluate(model, X, y):
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro"),
        "recall_macro": recall_score(y, y_pred, average="macro"),
        "f1_macro": f1_score(y, y_pred, average="macro")
    }

    return metrics, y_pred


# ==============================================
#        Plot Curves (FIXED)
# ==============================================
def generate_plots(model, X, y):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ❗ PyFunc does NOT support predict_proba
    # So we simulate probability using predictions (not ideal but works)
    y_pred = model.predict(X)

    classes = sorted(list(set(y)))
    y_bin = label_binarize(y, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    pr_path  = os.path.join(PLOTS_DIR, "pr_curve.png")

    # ROC Curve
    plt.figure()
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(roc_path)
    plt.close()

    # Precision-Recall Curve
    plt.figure()
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_bin[:, i])
        plt.plot(recall, precision, label=f"Class {classes[i]}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(pr_path)
    plt.close()

    return roc_path, pr_path


# ==============================================
#        Save Metrics Locally
# ==============================================
def save_metrics(metrics):
    os.makedirs("artifacts", exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Metrics saved locally at {METRICS_PATH}")


# ==============================================
#              Main
# ==============================================
def main():
    try:
        logging.info("=" * 60)
        logging.info("Starting Evaluation")
        logging.info("=" * 60)

        setup_mlflow("ayush-padaniya", "CyberGuard-url-detection-system")

        run_id = load_run_id()
        logging.info(f"Using Run ID: {run_id}")

        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        X, y = load_data()

        metrics, y_pred = evaluate(model, X, y)

        logging.info(f"Metrics: {metrics}")

        if metrics["f1_macro"] >= F1_THRESHOLD:
            logging.info("Model passed threshold")

            # Generate plots
            roc_path, pr_path = generate_plots(model, X, y)

            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(roc_path, artifact_path="plots")
                mlflow.log_artifact(pr_path, artifact_path="plots")

            save_metrics(metrics)

        else:
            logging.warning("Model did NOT pass threshold")

        logging.info("Evaluation Completed")

    except Exception as e:
        logging.error(f"Evaluation Failed: {e}")
        raise


if __name__ == "__main__":
    main()