import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
#          MLflow + DagsHub Setup
# ══════════════════════════════════════════════
dagshub.init(
    repo_owner='ayush-padaniya',
    repo_name='CyberGuard-url-detection-system',
    mlflow=True
)
mlflow.set_tracking_uri('https://dagshub.com/ayush-padaniya/CyberGuard-url-detection-system.mlflow')
mlflow.set_experiment("cyberguard-url-detection")

# ══════════════════════════════════════════════
#                Constants
# ══════════════════════════════════════════════
TRAIN_PATH = 'data/raw/train.csv'
TEST_PATH  = 'data/raw/test.csv'

COLUMNS_TO_DROP = ['url', 'domain', 'scan_date', 'type']
TARGET_COLUMN   = 'label'
RANDOM_STATE    = 42

# ══════════════════════════════════════════════
#              Load Data
# ══════════════════════════════════════════════
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# ══════════════════════════════════════════════
#             Preprocessing
# ══════════════════════════════════════════════
print("\nPreprocessing...")

# Step 1 — Drop unnecessary columns
train_df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')
test_df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')
print(f"✅ Dropped columns: {COLUMNS_TO_DROP}")

# Step 2 — Handle infinite values
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"✅ Infinite values handled")

# Step 3 — Handle missing values
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)
print(f"✅ Missing values handled")

# Step 4 — Remove duplicates
before = train_df.shape[0]
train_df.drop_duplicates(inplace=True)
after = train_df.shape[0]
print(f"✅ Removed {before - after} duplicate rows")

# Step 5 — Separate features and target
X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]
X_test  = test_df.drop(columns=[TARGET_COLUMN])
y_test  = test_df[TARGET_COLUMN]
print(f"✅ X_train: {X_train.shape}, y_train: {y_train.shape}")

# Step 6 — Scale features
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print(f"✅ Scaling completed")

print(f"\nClass distribution:\n{y_train.value_counts()}")

# ══════════════════════════════════════════════
#           Evaluation Function
# ══════════════════════════════════════════════
def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics dictionary."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall':    recall_score(y_test, y_pred, average='weighted'),
        'f1_score':  f1_score(y_test, y_pred, average='weighted'),
        'auc':       roc_auc_score(y_test, y_pred_prob,
                                   multi_class='ovr',
                                   average='weighted')
    }
    return metrics

# ══════════════════════════════════════════════
#            Models Definition
# ══════════════════════════════════════════════
models = {

    "Logistic Regression": {
        "model": LogisticRegression(
            C=1,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "params": {
            "model_name":   "Logistic Regression",
            "C":            1,
            "solver":       "lbfgs",
            "max_iter":     1000,
            "class_weight": "balanced"
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "params": {
            "model_name":   "Random Forest",
            "n_estimators": 100,
            "max_depth":    10,
            "class_weight": "balanced"
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        "params": {
            "model_name":    "Gradient Boosting",
            "n_estimators":  100,
            "learning_rate": 0.1,
            "max_depth":     5
        }
    },

    "XGBoost": {
        "model": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=3,
            random_state=RANDOM_STATE,
            tree_method='hist',
            device='cuda',
            eval_metric='mlogloss'
        ),
        "params": {
            "model_name":       "XGBoost",
            "n_estimators":     100,
            "learning_rate":    0.1,
            "max_depth":        6,
            "scale_pos_weight": 3,
            "tree_method":      "hist",
            "device":           "cuda"
        }
    }
}

# ══════════════════════════════════════════════
#     Train + Log Each Model in MLflow
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Starting MLflow Experiments...")
print("=" * 60)

for model_name, model_info in models.items():
    print(f"\n→ Training: {model_name}")

    with mlflow.start_run(run_name=model_name):
        try:
            # ── Train model ──
            model = model_info["model"]
            model.fit(X_train, y_train)
            print(f"  ✅ Training completed")

            # ── Evaluate model ──
            metrics = evaluate_model(model, X_test, y_test)
            print(f"  📊 Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  📊 F1 Score:  {metrics['f1_score']:.4f}")
            print(f"  📊 AUC:       {metrics['auc']:.4f}")

            # ── Log params to MLflow ──
            for param_name, param_value in model_info["params"].items():
                mlflow.log_param(param_name, param_value)

            # ── Log metrics to MLflow ──
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # ── Log model to MLflow ──
            mlflow.sklearn.log_model(model, "model")
            print(f"  ✅ Logged to MLflow successfully")

        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
            mlflow.end_run(status="FAILED")
            continue

print("\n" + "=" * 60)
print("✅ All Experiments Completed!")
print("🔗 Check: https://dagshub.com/ayush-padaniya/CyberGuard-url-detection-system.mlflow")
print("=" * 60)