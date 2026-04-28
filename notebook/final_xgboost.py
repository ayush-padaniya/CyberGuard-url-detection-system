import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
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
mlflow.set_experiment("cyberguard-final-model")

# ══════════════════════════════════════════════
#                Constants
# ══════════════════════════════════════════════
TRAIN_PATH      = 'data/raw/train.csv'
TEST_PATH       = 'data/raw/test.csv'
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

# ══════════════════════════════════════════════
#        Hyperparameter Tuning Grid
# ══════════════════════════════════════════════
print("\nSetting up Hyperparameter Tuning...")

param_grid = {
    'n_estimators':     [200, 300],
    'learning_rate':    [0.05, 0.1],
    'max_depth':        [6, 8],
    'min_child_weight': [1, 3],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Base XGBoost model with GPU
base_model = XGBClassifier(
    scale_pos_weight=3,
    random_state=RANDOM_STATE,
    tree_method='hist',
    device='cuda',
    eval_metric='mlogloss'
)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

# ══════════════════════════════════════════════
#         Train + Tune + Log to MLflow
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Starting Hyperparameter Tuning...")
print("=" * 60)

# End any active run
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="XGBoost_Final_Tuned"):
    try:
        # ── Run GridSearchCV ──
        print("Running GridSearchCV — this may take a while...")
        grid_search.fit(X_train, y_train)

        # ── Best params ──
        best_params = grid_search.best_params_
        best_model  = grid_search.best_estimator_
        print(f"\n✅ Best Parameters Found:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")

        # ── Evaluate best model ──
        y_pred      = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)

        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall':    recall_score(y_test, y_pred, average='weighted'),
            'f1_score':  f1_score(y_test, y_pred, average='weighted'),
            'auc':       roc_auc_score(y_test, y_pred_prob,
                                       multi_class='ovr',
                                       average='weighted')
        }

        print(f"\n📊 Final Model Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        print(f"   AUC:       {metrics['auc']:.4f}")

        # ── Log best params to MLflow ──
        mlflow.log_param("model_name",       "XGBoost Final Tuned")
        mlflow.log_param("tree_method",      "hist")
        mlflow.log_param("device",           "cuda")
        mlflow.log_param("scale_pos_weight", 3)
        mlflow.log_param("cv_folds",         3)
        mlflow.log_param("best_cv_score",    grid_search.best_score_)
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # ── Log metrics to MLflow ──
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # ── Log best model to MLflow ──
        mlflow.sklearn.log_model(best_model, "model")
        print(f"\n✅ Final model logged to MLflow successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        mlflow.end_run(status="FAILED")
        raise

print("\n" + "=" * 60)
print("✅ Final Model Training Completed!")
print("🔗 Check: https://dagshub.com/ayush-padaniya/CyberGuard-url-detection-system.mlflow")
print("=" * 60)