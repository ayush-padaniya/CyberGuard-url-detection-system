FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/app-requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/app-requirements.txt

COPY CyberGuard-Url-App ./CyberGuard-Url-App
COPY src ./src
COPY artifacts/preprocessing.pkl ./artifacts/preprocessing.pkl
COPY artifacts/metrics.json ./artifacts/metrics.json
COPY monitoring ./monitoring
COPY data/preprocessed ./data/preprocessed

RUN mkdir -p monitoring/reports

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "CyberGuard-Url-App", "--host", "0.0.0.0", "--port", "8000"]