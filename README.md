# Heart Disease Risk Prediction — End-to-End MLOps

Predicts heart-disease presence from 13 clinical attributes (UCI Heart Disease, Cleveland subset)
and serves it as a monitored, containerised API on Kubernetes.

| Stage | Where |
|---|---|
| Data download + cleaning | [scripts/download_data.py](scripts/download_data.py), [heartml/data.py](heartml/data.py) |
| EDA (figures in `reports/figures`) | [heartml/eda.py](heartml/eda.py), [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) |
| Preprocessing pipeline | [heartml/features.py](heartml/features.py) |
| Training, tuning, MLflow tracking | [heartml/train.py](heartml/train.py), [notebooks/02_training.ipynb](notebooks/02_training.ipynb) |
| Inference | [heartml/predict.py](heartml/predict.py), [notebooks/03_inference.ipynb](notebooks/03_inference.ipynb) |
| API (FastAPI + Prometheus metrics) | [api/main.py](api/main.py) |
| Unit tests | [tests/](tests/) |
| CI/CD | [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| Container | [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) |
| Kubernetes | [k8s/api.yaml](k8s/api.yaml), [k8s/monitoring.yaml](k8s/monitoring.yaml), [scripts/deploy_k8s.sh](scripts/deploy_k8s.sh) |
| Report | [reports/REPORT.md](reports/REPORT.md) |

## 1. Setup

Python 3.12+ (tested on 3.12 and 3.14).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the pipeline

```bash
python scripts/download_data.py      # data/raw + data/processed/heart_clean.csv
python -m heartml.eda                # reports/figures/*.png, reports/eda_summary.json
python -m heartml.train              # tunes 3 models, logs to MLflow, saves models/heart_model.joblib
python -m heartml.train --quick      # tiny grids, ~20 s
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
pytest -v
flake8 heartml api tests scripts
```

## 3. Serve locally

```bash
uvicorn api.main:app --port 8000          # Swagger UI at http://localhost:8000/docs
python scripts/smoke_test.py --url http://127.0.0.1:8000
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/patient.json
```
```json
{"prediction":1,"label":"heart disease","probability_disease":0.8793,"confidence":0.8793,
 "risk_level":"high","model_name":"logistic_regression","request_id":"c764470a9ad4"}
```

Endpoints: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`, `GET /metrics`.

## 4. Docker

```bash
docker build -t heart-api:1.0.0 .            # needs models/heart_model.joblib (train first)
docker run -p 8000:8000 heart-api:1.0.0
python scripts/smoke_test.py --url http://127.0.0.1:8000

docker compose up --build                    # API + Prometheus (9090) + Grafana (3000, admin/admin)
python scripts/smoke_test.py --load 300      # traffic for the dashboard
```

## 5. Kubernetes (Docker Desktop or Minikube)

```bash
bash scripts/deploy_k8s.sh
```

| | Docker Desktop | Minikube |
|---|---|---|
| API (LoadBalancer) | http://localhost/health | `minikube tunnel`, then the service EXTERNAL-IP |
| Ingress (`heart-api.local`) | needs ingress-nginx installed | `minikube addons enable ingress`; add `heart-api.local` to hosts file |
| Prometheus | http://localhost:9090 | `minikube service prometheus -n heart-mlops` |
| Grafana | http://localhost:3000 | `minikube service grafana -n heart-mlops` |

```bash
kubectl -n heart-mlops get pods,svc,ingress
kubectl -n heart-mlops logs -l app=heart-api --tail=20      # JSON request logs
python scripts/smoke_test.py --url http://127.0.0.1 --load 300
```

Tear down: `kubectl delete namespace heart-mlops`.

## Results (held-out 20% test split, 5-fold CV on the training 80%)

| Model | CV ROC-AUC | CV accuracy | Test accuracy | Test precision | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** (selected) | **0.905** | 0.847 | 0.885 | 0.839 | 0.929 | 0.969 |
| Random Forest | 0.899 | 0.830 | 0.885 | 0.839 | 0.929 | 0.949 |
| Gradient Boosting | 0.884 | 0.818 | 0.902 | 0.867 | 0.929 | 0.960 |
