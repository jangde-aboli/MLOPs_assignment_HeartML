# Screenshots

Save each screenshot here with the exact filename below; `reports/REPORT.md` embeds them.
Then rebuild the report: `bash scripts/build_report.sh`.

| File | What to capture | How |
|---|---|---|
| `mlflow_runs.png` | Experiment table comparing the runs | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → experiment *heart-disease-classification*, add metric columns `cv_roc_auc_mean`, `test_accuracy` |
| `mlflow_run_detail.png` | One run: params, metrics, `plots/` artifacts | Click the `logistic_regression` run |
| `github_actions_run.png` | A green run with lint → test → train → docker | GitHub → Actions → latest run |
| `github_actions_artifacts.png` | Artifacts section of that run | Bottom of the run summary page |
| `docker_build_run.png` | `docker build` finishing + `docker run` + a curl to `/predict` | Terminal |
| `k8s_pods_services.png` | `kubectl -n heart-mlops get pods,svc,ingress` | Terminal, after `scripts/deploy_k8s.sh` |
| `k8s_predict.png` | `python scripts/smoke_test.py --url http://127.0.0.1` passing | Terminal |
| `grafana_dashboard.png` | *Heart Disease API* dashboard with traffic | Run `smoke_test.py --load 300` first, open Grafana (http://localhost:3000) |
| `prometheus_targets.png` | Status → Targets, the API pods UP | Prometheus (http://localhost:9090) |
| `api_logs.png` | JSON request logs | `kubectl -n heart-mlops logs -l app=heart-api --tail=20` |
