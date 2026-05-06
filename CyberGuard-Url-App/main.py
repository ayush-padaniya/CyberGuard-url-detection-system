import os
import sys
import json
import joblib
import ipaddress
import re
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.xgboost
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from src.logger import logging
from fastapi.responses import FileResponse
from monitoring.evidently_monitor import run_evidently_report

# ══════════════════════════════════════════════
#         MLflow + DagsHub Setup
# ══════════════════════════════════════════════
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
mlflow.set_tracking_uri(
    "https://dagshub.com/ayush-padaniya/CyberGuard-url-detection-system.mlflow"
)
# ══════════════════════════════════════════════
#         Constants
# ══════════════════════════════════════════════
MODEL_NAME       = "CyberGuard-XGBoost"
MODEL_ALIAS      = "Production"
BASE_DIR         = Path(__file__).resolve().parent
PROJECT_ROOT     = BASE_DIR.parent
SCALER_PATH      = PROJECT_ROOT / "artifacts" / "preprocessing.pkl"
METRICS_PATH     = PROJECT_ROOT / "artifacts" / "metrics.json"

LABEL_MAP = {
    0: {"name": "Benign",      "color": "#00ff88", "icon": "✅", "risk": "Safe"},
    1: {"name": "Defacement",  "color": "#ff9900", "icon": "⚠️", "risk": "Medium"},
    2: {"name": "Phishing",    "color": "#ff4444", "icon": "🎣", "risk": "High"},
    3: {"name": "Malware",     "color": "#ff0000", "icon": "☠️", "risk": "Critical"},
}

FEATURE_COLUMNS = [
    'url_len', '@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',', '//',
    'digits', 'letters', 'abnormal_url', 'https', 'Shortining_Service',
    'having_ip_address', 'web_http_status', 'web_is_live', 'web_ext_ratio',
    'web_unique_domains', 'web_favicon', 'web_csp', 'web_xframe', 'web_hsts',
    'web_xcontent', 'web_security_score', 'web_forms_count', 'web_password_fields',
    'web_hidden_inputs', 'web_has_login', 'web_ssl_valid', 'phish_urgency_words',
    'phish_security_words', 'phish_brand_mentions', 'phish_brand_hijack',
    'phish_multiple_subdomains', 'phish_long_path', 'phish_many_params',
    'phish_suspicious_tld', 'phish_adv_exact_brand_match', 'phish_adv_brand_in_subdomain',
    'phish_adv_brand_in_path', 'phish_adv_hyphen_count', 'phish_adv_number_count',
    'phish_adv_suspicious_tld', 'phish_adv_long_domain', 'phish_adv_many_subdomains',
    'phish_adv_encoded_chars', 'phish_adv_path_keywords', 'phish_adv_has_redirect',
    'phish_adv_many_params', 'path_has_hacked_terms', 'suspicious_extension',
    'path_underscore_count', 'is_gov_edu',
    'digit_letter_ratio', 'special_char_ratio', 'url_complexity'
]

# ══════════════════════════════════════════════
#         Prometheus Metrics
# ══════════════════════════════════════════════
REQUEST_COUNT = Counter(
    'cyberguard_request_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

PREDICTION_COUNT = Counter(
    'cyberguard_prediction_total',
    'Total predictions by threat type',
    ['threat_type', 'risk_level']
)

REQUEST_LATENCY = Histogram(
    'cyberguard_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

MODEL_CONFIDENCE = Histogram(
    'cyberguard_model_confidence',
    'Model prediction confidence distribution',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

TOTAL_PREDICTIONS = Gauge(
    'cyberguard_total_predictions',
    'Total number of predictions made so far'
)

THREAT_GAUGE = Gauge(
    'cyberguard_threat_detected',
    'Last threat detected (0=Benign, 1=Defacement, 2=Phishing, 3=Malware)'
)

# ══════════════════════════════════════════════
#         Load Model + Scaler
# ══════════════════════════════════════════════
def load_registered_model():
    logging.info("Loading model from MLflow registry...")
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    model_uri = f"models:/{MODEL_NAME}/{model_version}"

    try:
        loaded_model = mlflow.xgboost.load_model(model_uri)
        logging.info(f"Loaded XGBoost flavor: {MODEL_NAME} v{model_version}")
        return loaded_model, model_version, True
    except Exception as e:
        logging.warning(f"XGBoost flavor unavailable, loading pyfunc flavor instead: {e}")
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"Loaded pyfunc flavor: {MODEL_NAME} v{model_version}")
        return loaded_model, model_version, False


model, version, supports_predict_proba = load_registered_model()

logging.info("Loading scaler...")
scaler = joblib.load(SCALER_PATH)
logging.info("Scaler loaded successfully")

# ══════════════════════════════════════════════
#         Load Metrics
# ══════════════════════════════════════════════
def load_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except:
        return {}

# ══════════════════════════════════════════════
#         Feature Extraction
# ══════════════════════════════════════════════
def extract_features(url: str) -> dict:
    features = {col: 0 for col in FEATURE_COLUMNS}
    normalized_url = url.strip()
    parsed = urlparse(normalized_url if "://" in normalized_url else f"http://{normalized_url}")
    hostname = (parsed.hostname or "").lower()
    domain_parts = [part for part in hostname.split(".") if part]
    path = parsed.path.lower()
    query = parsed.query.lower()
    full_lower = normalized_url.lower()

    features['url_len']  = len(normalized_url)
    features['digits']   = sum(c.isdigit() for c in normalized_url)
    features['letters']  = sum(c.isalpha() for c in normalized_url)

    for char in ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',']:
        features[char] = normalized_url.count(char)
    features['//'] = normalized_url.count('//')

    features['https'] = 1 if parsed.scheme == "https" else 0
    features['abnormal_url'] = 1 if "@" in normalized_url else 0
    features['Shortining_Service'] = 1 if any(
        service in hostname
        for service in ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']
    ) else 0

    try:
        ipaddress.ip_address(hostname)
        features['having_ip_address'] = 1
    except ValueError:
        features['having_ip_address'] = 0

    if features['https']:
        features['web_http_status'] = 200
        features['web_is_live'] = 1
        features['web_ssl_valid'] = 1
        features['web_security_score'] = 70
        features['web_hsts'] = 1
        features['web_xcontent'] = 1
    elif parsed.scheme == "http":
        features['web_http_status'] = 200
        features['web_is_live'] = 1
        features['web_security_score'] = 25

    features['web_unique_domains'] = max(len(set(domain_parts)) - 1, 0)

    urgency_words = ['urgent', 'verify', 'limited', 'suspended', 'locked', 'expire', 'immediately']
    security_words = ['login', 'signin', 'account', 'password', 'secure', 'security', 'bank', 'wallet']
    brands = ['paypal', 'google', 'amazon', 'microsoft', 'apple', 'facebook', 'instagram', 'netflix']
    suspicious_tlds = ['zip', 'xyz', 'top', 'click', 'link', 'ru', 'cn', 'tk', 'ml', 'ga', 'cf', 'gq']
    hacked_terms = ['hacked', 'defaced', 'owned', 'pwned', 'shell', 'adminer']
    executable_exts = ['.exe', '.scr', '.bat', '.cmd', '.msi', '.apk', '.jar', '.zip', '.rar']
    redirect_words = ['redirect', 'redir', 'url=', 'next=', 'return=', 'continue=']

    tld = domain_parts[-1] if domain_parts else ""
    subdomain_count = max(len(domain_parts) - 2, 0)
    hyphen_count = hostname.count("-")
    domain_digit_count = sum(c.isdigit() for c in hostname)
    params = parse_qs(parsed.query)

    features['phish_urgency_words'] = sum(word in full_lower for word in urgency_words)
    features['phish_security_words'] = sum(word in full_lower for word in security_words)
    features['phish_brand_mentions'] = sum(brand in full_lower for brand in brands)
    features['phish_brand_hijack'] = 1 if features['phish_brand_mentions'] and hyphen_count else 0
    features['phish_multiple_subdomains'] = 1 if subdomain_count >= 2 else 0
    features['phish_long_path'] = 1 if len(path) > 60 else 0
    features['phish_many_params'] = 1 if len(params) >= 3 else 0
    features['phish_suspicious_tld'] = 1 if tld in suspicious_tlds else 0
    features['phish_adv_exact_brand_match'] = 1 if hostname in {f"{brand}.com" for brand in brands} else 0
    features['phish_adv_brand_in_subdomain'] = 1 if any(
        brand in ".".join(domain_parts[:-2]) for brand in brands
    ) else 0
    features['phish_adv_brand_in_path'] = 1 if any(brand in path for brand in brands) else 0
    features['phish_adv_hyphen_count'] = hyphen_count
    features['phish_adv_number_count'] = domain_digit_count
    features['phish_adv_suspicious_tld'] = features['phish_suspicious_tld']
    features['phish_adv_long_domain'] = 1 if len(hostname) > 35 else 0
    features['phish_adv_many_subdomains'] = 1 if subdomain_count >= 3 else 0
    features['phish_adv_encoded_chars'] = full_lower.count('%')
    features['phish_adv_path_keywords'] = sum(word in path for word in security_words + hacked_terms)
    features['phish_adv_has_redirect'] = 1 if any(word in query for word in redirect_words) else 0
    features['phish_adv_many_params'] = features['phish_many_params']
    features['path_has_hacked_terms'] = sum(word in path for word in hacked_terms)
    features['suspicious_extension'] = 1 if any(path.endswith(ext) for ext in executable_exts) else 0
    features['path_underscore_count'] = path.count('_')
    features['is_gov_edu'] = 1 if tld in ['gov', 'edu'] else 0

    features['digit_letter_ratio'] = features['digits'] / (features['letters'] + 1)
    features['special_char_ratio'] = sum(features[c] for c in ['@','?','-','=','.','#','%','+','$','!','*',',','//']) / (features['url_len'] + 1)
    features['url_complexity']     = features['url_len'] * (features['digits'] + 1)

    return features


def predict_with_confidence(input_df: pd.DataFrame):
    prediction = model.predict(input_df)
    pred_value = prediction[0] if hasattr(prediction, "__len__") else prediction

    if supports_predict_proba and hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(input_df)[0]
        confidence = round(float(max(pred_proba)) * 100, 2)
    else:
        confidence = 100.0

    return int(pred_value), confidence


def classify_with_url_policy(features: dict):
    suspicious_score = (
        features['abnormal_url']
        + features['Shortining_Service']
        + features['having_ip_address']
        + features['phish_urgency_words']
        + features['phish_security_words']
        + features['phish_brand_hijack']
        + features['phish_multiple_subdomains']
        + features['phish_many_params']
        + features['phish_suspicious_tld']
        + features['phish_adv_brand_in_subdomain']
        + features['phish_adv_brand_in_path']
        + features['phish_adv_long_domain']
        + features['phish_adv_many_subdomains']
        + features['phish_adv_has_redirect']
    )

    if features['suspicious_extension']:
        return 3, 94.0
    if features['path_has_hacked_terms']:
        return 1, 92.0
    if (
        features['https']
        and features['web_ssl_valid']
        and features['url_len'] <= 120
        and suspicious_score == 0
        and features['digits'] <= 5
        and features['phish_adv_hyphen_count'] <= 2
    ):
        return 0, 93.0
    if suspicious_score >= 2:
        return 2, 91.0

    return None, None

# ══════════════════════════════════════════════
#         FastAPI App
# ══════════════════════════════════════════════
app = FastAPI(title="CyberGuard URL Detection")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

prediction_history = []

# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    start_time = time.time()
    metrics = load_metrics()
    REQUEST_COUNT.labels(method="GET", endpoint="/", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return templates.TemplateResponse(request, "index.html", {
        "metrics": metrics
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, url: str = Form(...)):
    start_time = time.time()
    try:
        features = extract_features(url)
        df       = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        pred, confidence = classify_with_url_policy(features)
        if pred is None:
            df_scaled = scaler.transform(df)
            df_scaled = pd.DataFrame(df_scaled, columns=FEATURE_COLUMNS)
            pred, confidence = predict_with_confidence(df_scaled)

        label_info = LABEL_MAP[pred]

        # ── Update Prometheus metrics ──
        PREDICTION_COUNT.labels(
            threat_type=label_info["name"],
            risk_level=label_info["risk"]
        ).inc()
        MODEL_CONFIDENCE.observe(confidence / 100)
        THREAT_GAUGE.set(pred)
        TOTAL_PREDICTIONS.inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        prediction_history.append({
            "url":        url[:50] + "..." if len(url) > 50 else url,
            "prediction": label_info["name"],
            "confidence": confidence,
            "risk":       label_info["risk"],
            "risk_class": label_info["risk"].lower(),
            "color":      label_info["color"]
        })

        if len(prediction_history) > 10:
            prediction_history.pop(0)

        return templates.TemplateResponse(request, "index.html", {
            "url":        url,
            "prediction": label_info["name"],
            "confidence": confidence,
            "color":      label_info["color"],
            "icon":       label_info["icon"],
            "risk":       label_info["risk"],
            "risk_class": label_info["risk"].lower(),
            "metrics":    load_metrics(),
            "show_result": True
        })

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        logging.error(f"Prediction error: {e}")
        return templates.TemplateResponse(request, "index.html", {
            "error":   str(e),
            "metrics": load_metrics()
        })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    start_time = time.time()
    metrics = load_metrics()
    performance_values = [
        round(metrics.get("accuracy", 0) * 100, 1),
        round(metrics.get("f1_macro", 0) * 100, 1),
        round(metrics.get("precision_macro", 0) * 100, 1),
        round(metrics.get("recall_macro", 0) * 100, 1),
    ]
    REQUEST_COUNT.labels(method="GET", endpoint="/dashboard", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="/dashboard").observe(time.time() - start_time)
    return templates.TemplateResponse(request, "dashboard.html", {
        "metrics":            metrics,
        "performance_values": performance_values,
        "prediction_history": prediction_history,
        "total_predictions":  len(prediction_history),
        "model_name":         MODEL_NAME,
        "model_version":      version
    })


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  model is not None,
        "scaler_loaded": scaler is not None
    }


@app.get("/model-info")
async def model_info():
    return {
        "model_name":             MODEL_NAME,
        "model_alias":            MODEL_ALIAS,
        "model_version":          version,
        "supports_predict_proba": supports_predict_proba
    }


@app.get("/api/history")
async def get_history():
    return {"history": prediction_history}

@app.get("/monitoring/report")
async def evidently_report():
    report_path = "monitoring/reports/evidently_report.html"
    os.makedirs("monitoring/reports", exist_ok=True)
    if not os.path.exists(report_path):
        run_evidently_report()
    return FileResponse(report_path, media_type="text/html")

# ── Prometheus metrics endpoint ──
@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )