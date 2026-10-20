"""FastAPI service for heart-disease risk prediction.

Endpoints:
    GET  /health         liveness/readiness probe
    GET  /model-info     metadata of the loaded model
    POST /predict        one patient -> prediction + confidence
    POST /predict/batch  up to 100 patients
    GET  /metrics        Prometheus metrics
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)

from api.schemas import (
    BatchPrediction, BatchRequest, PatientFeatures, PredictResponse,
)
from heartml import config
from heartml.predict import load_model, predict_records


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so logs are easy to grep or ship to Loki/ELK."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                 "level": record.levelname, "logger": record.name,
                 "msg": record.getMessage()}
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("heart_api")
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        log_file = os.getenv("LOG_FILE")
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(JsonFormatter())
            logger.addHandler(fh)
    logger.propagate = False
    return logger


log = _setup_logging()

REQUESTS = Counter("heart_api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency",
                    ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions by class",
                      ["label"])
PROBABILITY = Histogram("heart_api_disease_probability",
                        "Distribution of predicted disease probability",
                        buckets=[i / 10 for i in range(1, 11)])
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if a model is loaded")

state: dict = {}


def _load() -> None:
    path = Path(os.getenv("MODEL_PATH", config.MODEL_FILE))
    state["model"] = load_model(path)
    meta_path = path.with_name("model_meta.json")
    state["meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    MODEL_LOADED.set(1)
    log.info("model loaded", extra={"extra_fields": {
        "path": str(path), "model_name": state["meta"].get("model_name")}})


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _load()
    except FileNotFoundError as err:
        MODEL_LOADED.set(0)
        log.error(str(err))
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        # Use the route template (/predict) rather than the raw path to keep
        # metric label cardinality bounded.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            log.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method,
                "path": request.url.path, "status": status,
                "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


def _model():
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    return state["model"]


def _record_predictions(results: list[dict]) -> None:
    for r in results:
        PREDICTIONS.labels(r["label"]).inc()
        PROBABILITY.observe(r["probability_disease"])


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "degraded",
            "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    _model()
    return state["meta"]


@app.post("/predict", response_model=PredictResponse)
def predict(patient: PatientFeatures, request: Request):
    result = predict_records(_model(), [patient.model_dump()])[0]
    _record_predictions([result])
    log.info("prediction", extra={"extra_fields": {
        "request_id": request.state.request_id, "input": patient.model_dump(),
        **result}})
    return {**result, "model_name": state["meta"].get("model_name", "unknown"),
            "request_id": request.state.request_id}


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(batch: BatchRequest):
    results = predict_records(_model(), [p.model_dump() for p in batch.patients])
    _record_predictions(results)
    return {"predictions": results,
            "model_name": state["meta"].get("model_name", "unknown")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
