# Serving image for the heart-disease API.
# Expects models/heart_model.joblib to exist (run `python -m heartml.train` first;
# CI does this before `docker build`).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/heart_model.joblib

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

COPY heartml/ heartml/
COPY api/ api/
COPY models/heart_model.joblib models/model_meta.json models/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/logs && chown -R appuser /app
# Numeric UID so Kubernetes runAsNonRoot can verify it.
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Single worker: prometheus_client keeps metrics per process, so scale with
# container replicas (Kubernetes) rather than uvicorn workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
