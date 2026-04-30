# CyberGuard URL Detection System

End-to-end MLOps project for malicious URL detection with DVC pipelines,
MLflow/DagsHub model registry, and a FastAPI inference app.

## Run App Locally

From the project root:

```powershell
.\cyber_guard\Scripts\python.exe -m uvicorn main:app --app-dir CyberGuard-Url-App --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Docker

Start Docker Desktop first, then build the image from the project root:

```powershell
docker build -t cyberguard-url-app:latest .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 cyberguard-url-app:latest
```

Open:

```text
http://127.0.0.1:8000
```

Useful API checks:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/model-info
```

If your DagsHub/MLflow model requires authentication inside Docker, pass the
token as an environment variable:

```powershell
docker run --rm -p 8000:8000 -e DAGSHUB_USER_TOKEN="your_token_here" cyberguard-url-app:latest
```
