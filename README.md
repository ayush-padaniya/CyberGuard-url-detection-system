# 🛡️ CyberGuard — URL Threat Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange?style=for-the-badge&logo=xgboost&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20ECR%20%7C%20S3-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?style=for-the-badge&logo=grafana&logoColor=white)

<br/>

**A production-grade MLOps system that detects malicious URLs in real-time using XGBoost, deployed on AWS with full CI/CD automation, monitoring, and alerting.**

<br/>

[🌐 Live Demo](http://16.16.122.170:8000) · [📊 Grafana Dashboard](http://16.16.122.170:3001) · [📈 Prometheus](http://16.16.122.170:9090)

</div>

---

## 📌 Table of Contents

- [Business Problem](#-business-problem)
- [Objective](#-objective)
- [Solution](#-solution)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [MLOps Architecture](#-mlops-architecture)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Monitoring](#-monitoring)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Screenshots](#-screenshots)
- [Getting Started](#-getting-started)
- [API Endpoints](#-api-endpoints)
- [Author](#-author)

---

## 💼 Business Problem

Cyber threats through malicious URLs are one of the fastest-growing security challenges worldwide. Every day, millions of users unknowingly click on phishing links, malware-infected URLs, and defacement attacks that lead to:

- **Financial losses** — banking credentials stolen via phishing attacks
- **Data breaches** — malware installed through drive-by downloads
- **Brand damage** — website defacement destroying company reputation
- **Identity theft** — personal information harvested through fake login pages

Traditional rule-based URL filters fail to keep up with increasingly sophisticated attacks. Security teams need **intelligent, real-time threat detection** that can classify URLs instantly with high accuracy.

---

## 🎯 Objective

Build a **production-ready URL threat detection system** that:

- Classifies any URL into 4 threat categories in real-time
- Achieves high accuracy using machine learning
- Provides a user-friendly interface for non-technical users
- Monitors model performance and data drift continuously
- Deploys automatically on every code change via CI/CD

---

## ✅ Solution

CyberGuard is an end-to-end MLOps system that:

1. **Extracts 57 features** from raw URLs (structural, lexical, security signals)
2. **Classifies threats** using XGBoost trained on 651,192 URLs
3. **Deploys via FastAPI** with a beautiful dark cybersecurity UI
4. **Monitors in production** using Prometheus, Grafana, and Evidently AI
5. **Auto-deploys** to AWS EC2 on every GitHub push

### Threat Categories

| Label | Class | Description |
|-------|-------|-------------|
| 0 | ✅ Benign | Safe, legitimate URL |
| 1 | ⚠️ Defacement | Website content altered by attacker |
| 2 | 🎣 Phishing | Designed to steal credentials |
| 3 | ☠️ Malware | Installs malicious software |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **89.25%** |
| Precision (Macro) | **84.26%** |
| Recall (Macro) | **82.43%** |
| F1 Score (Macro) | **83.23%** |

- **Dataset:** 651,192 URLs (~200MB)
- **Algorithm:** XGBoost Classifier
- **Features:** 57 engineered URL features
- **Validation:** Leakage-safe URL-based group split

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11 |
| **ML Framework** | XGBoost, Scikit-learn |
| **Experiment Tracking** | MLflow, DagsHub |
| **Data Versioning** | DVC + AWS S3 |
| **Data Validation** | Great Expectations |
| **Web Framework** | FastAPI, Jinja2 |
| **Containerization** | Docker, Docker Compose |
| **Container Registry** | AWS ECR |
| **Cloud Deployment** | AWS EC2 |
| **CI/CD** | GitHub Actions |
| **Metrics Monitoring** | Prometheus |
| **Dashboard** | Grafana |
| **Drift Detection** | Evidently AI |
| **Model Registry** | MLflow Model Registry |

---

## 🏗️ MLOps Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEVELOPER                                 │
│                    git push → main                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GITHUB ACTIONS CI/CD                           │
│                                                                  │
│  Install deps → DVC Pull → DVC Repro → Build Image → Push ECR  │
│                            ↓                                     │
│              SSH into EC2 → Pull Image → Deploy                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AWS EC2                                   │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  CyberGuard App │  │  Prometheus  │  │     Grafana      │   │
│  │  FastAPI :8000  │  │    :9090     │  │      :3001       │   │
│  │                 │◄─│  (scraping)  │◄─│   (dashboard)    │   │
│  │  XGBoost Model  │  │              │  │   (alerting)     │   │
│  │  Evidently AI   │  └──────────────┘  └──────────────────┘   │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                             │
│                                                                  │
│   DagsHub (MLflow)    AWS S3 (Data)    Gmail (Alerts)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
CyberGuard-url-detection-system/
│
├── .github/
│   └── workflows/
│       └── ci.yaml                  ← CI/CD Pipeline
│
├── CyberGuard-Url-App/
│   ├── main.py                      ← FastAPI App
│   ├── templates/
│   │   ├── index.html               ← Home Page
│   │   └── dashboard.html           ← Dashboard Page
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── src/
│   ├── logger/
│   ├── connections/
│   └── pipeline/
│       ├── Data_Ingestion.py
│       ├── Data_Validation.py
│       ├── Data_Preprocessing.py
│       ├── Feature_Engineering.py
│       ├── Model_Training.py
│       ├── Model_Evaluation.py
│       └── Model_Registry.py
│
├── monitoring/
│   ├── prometheus.yml               ← Prometheus Config
│   ├── evidently_monitor.py         ← Drift Detection
│   └── reports/                     ← Generated Reports
│
├── artifacts/
│   ├── preprocessing.pkl            ← Saved Scaler
│   ├── metrics.json                 ← Model Metrics
│   └── plots/                       ← ROC, PR Curves
│
├── data/                            ← DVC Tracked
│   ├── raw/
│   ├── processed/
│   └── preprocessed/
│
├── config/
│   └── config_mlflow.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── params.yaml
├── dvc.yaml
└── README.md
```

---

## 🔄 ML Pipeline

The pipeline is fully automated using DVC and runs on every push:

```
Data Ingestion
      ↓
  Download from S3
  URL Deduplication
  Leakage-safe Group Split (by URL)
      ↓
Data Validation (Great Expectations)
      ↓
  Row count check
  Required columns check
  No nulls validation
  Label value validation
  URL length range check
      ↓
Feature Engineering
      ↓
  57 URL features extracted
  digit_letter_ratio
  special_char_ratio
  url_complexity
      ↓
Data Preprocessing
      ↓
  StandardScaler (fit on train only)
  Scaler saved to artifacts/
      ↓
Model Training
      ↓
  XGBoost Classifier
  Logged to MLflow/DagsHub
  Run ID saved locally
      ↓
Model Evaluation
      ↓
  Accuracy, F1, Precision, Recall
  ROC Curve, PR Curve
  F1 threshold check (≥ 0.80)
      ↓
Model Registry
      ↓
  Registered in MLflow Model Registry
  Promoted to Production alias
```

---

## 📡 Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cyberguard_request_total` | Counter | Total API requests |
| `cyberguard_prediction_total` | Counter | Predictions by threat type |
| `cyberguard_request_latency_seconds` | Histogram | Request latency |
| `cyberguard_model_confidence` | Histogram | Model confidence scores |
| `cyberguard_total_predictions` | Gauge | Total predictions made |
| `cyberguard_threat_detected` | Gauge | Last threat detected |

### Grafana Dashboard (10 Panels)

- Total Requests
- Total Predictions
- Threat Distribution (Pie Chart)
- Average Request Latency
- Last Threat Detected
- Average Model Confidence
- Predictions Per Minute
- Threat Level Over Time
- Total Malware Detected
- Total Phishing Detected

### Grafana Alerting

- **Alert:** High Threat Alert
- **Condition:** `cyberguard_threat_detected > 1`
- **Notification:** Gmail email alert
- **Evaluation:** Every 1 minute

### Evidently AI

- **Data Drift Detection** — feature distribution shift
- **Target Drift Detection** — label distribution shift
- **Report:** Available at `/monitoring/report`

---

## ⚙️ CI/CD Pipeline

### CI (Continuous Integration)
Triggers on every push to `main`:

```
Checkout → Python Setup → Install Dependencies
→ AWS Credentials → DVC Pull → DVC Repro
→ ECR Login → Docker Build → Docker Tag → Push to ECR
```

### CD (Continuous Deployment)
Runs after CI succeeds:

```
SSH into EC2 → Set Env Vars → ECR Login
→ Git Pull → DVC Pull → Docker Pull → docker-compose up
```

---

## 📸 Screenshots

### 🏠 Home Page — URL Scanner
<!-- Add screenshot: app home page with URL input -->
![Home Page](assets/screenshots/home.png)

### 🔴 Result Page — Threat Detection
<!-- Add screenshot: result page showing phishing detection -->
![Result Page](assets/screenshots/result.png)

### 📊 Grafana Dashboard
<!-- Add screenshot: Grafana dashboard with all 10 panels -->
![Grafana Dashboard](assets/screenshots/grafana.png)

### 🚨 Alert Firing
<!-- Add screenshot: Grafana alert firing red -->
![Alert](assets/screenshots/alert.png)

### 📉 Evidently AI Report
<!-- Add screenshot: Evidently drift detection report -->
![Evidently Report](assets/screenshots/evidently.png)

### ✅ GitHub Actions CI/CD
<!-- Add screenshot: GitHub Actions green ticks for CI and CD -->
![CI/CD](assets/screenshots/cicd.png)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- AWS Account (ECR, EC2, S3)
- DagsHub Account
- Git

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/ayush-padaniya/CyberGuard-url-detection-system.git
cd CyberGuard-url-detection-system

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Set environment variables
export DAGSHUB_USERNAME="your_username"
export DAGSHUB_TOKEN="your_token"
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_REGION="your_region"
export S3_BUCKET_NAME="your_bucket"

# 4. Pull data with DVC
dvc pull

# 5. Run ML pipeline
dvc repro

# 6. Run app locally
cd CyberGuard-Url-App
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Setup

```bash
# 1. Set environment variables (see above)

# 2. Login to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <ecr-uri>

# 3. Pull image
docker pull <ecr-uri>/cyberguard-url-app:latest

# 4. Start all services
docker-compose up -d

# 5. Check logs
docker-compose logs -f cyberguard-app
```

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key |
| `AWS_REGION` | AWS Region |
| `AWS_ACCOUNT_ID` | AWS Account ID |
| `ECR_REGISTRY` | ECR Repository Name |
| `S3_BUCKET_NAME` | S3 Bucket Name |
| `DAGSHUB_USERNAME` | DagsHub Username |
| `DAGSHUB_TOKEN` | DagsHub Token |
| `EC2_HOST` | EC2 Public IP |
| `EC2_USER` | EC2 Username (ubuntu) |
| `EC2_SSH_KEY` | EC2 SSH Private Key |
| `GF_SECURITY_ADMIN_PASSWORD` | Grafana Admin Password |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page — URL scanner |
| `/predict` | POST | Predict threat for URL |
| `/dashboard` | GET | App dashboard |
| `/health` | GET | Health check |
| `/model-info` | GET | Model version info |
| `/metrics` | GET | Prometheus metrics |
| `/monitoring/report` | GET | Evidently AI drift report |
| `/api/history` | GET | Recent predictions |

---

## 🌐 Live URLs

| Service | URL |
|---------|-----|
| App | http://16.16.122.170:8000 |
| Grafana | http://16.16.122.170:3001 |
| Prometheus | http://16.16.122.170:9090 |
| Evidently | http://16.16.122.170:8000/monitoring/report |

---

<div align="center">

---

**Built with ❤️ by**

## Ayush Padaniya

[![GitHub](https://img.shields.io/badge/GitHub-ayush--padaniya-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ayush-padaniya)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_Padaniya-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ayush-padaniya)

</div>
