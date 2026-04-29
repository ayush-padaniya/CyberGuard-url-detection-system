import os
import mlflow

def setup_mlflow(repo_owner, repo_name):

    token = os.getenv("DAGSHUB_TOKEN")
    username = os.getenv("DAGSHUB_USERNAME")

    if not token or not username:
        raise ValueError("Missing DAGSHUB credentials")

    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

    # 🔥 IMPORTANT: set tracking URI FIRST
    mlflow.set_tracking_uri(tracking_uri)

    # 🔥 THEN set credentials in MLflow auth style
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    print("MLflow connected:", tracking_uri)