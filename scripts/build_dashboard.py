"""Generate the Grafana dashboard JSON for the heart-disease API."""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}


def panel(pid, title, exprs, x, y, w=12, h=8, kind="timeseries", unit=None, legend=None):
    p = {"id": pid, "title": title, "type": kind, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [{"refId": chr(65 + i), "expr": e, "datasource": DS,
                      **({"legendFormat": legend[i]} if legend else {})}
                     for i, e in enumerate(exprs)],
         "fieldConfig": {"defaults": {}, "overrides": []}, "options": {}}
    if unit:
        p["fieldConfig"]["defaults"]["unit"] = unit
    return p


panels = [
    panel(1, "Model loaded (all pods)", ["min(heart_api_model_loaded)"], 0, 0, 6, 4, "stat"),
    panel(2, "Total predictions", ["sum(heart_api_predictions_total)"], 6, 0, 6, 4, "stat"),
    panel(3, "Disease share of predictions",
          ['sum(heart_api_predictions_total{label="heart disease"}) / '
           'sum(heart_api_predictions_total)'], 12, 0, 6, 4, "stat", "percentunit"),
    panel(4, "Error rate (4xx+5xx)",
          ['sum(rate(heart_api_requests_total{status=~"4..|5.."}[1m])) / '
           'clamp_min(sum(rate(heart_api_requests_total[1m])), 1e-9)'],
          18, 0, 6, 4, "stat", "percentunit"),
    panel(5, "Request rate by endpoint",
          ["sum by (endpoint) (rate(heart_api_requests_total[1m]))"], 0, 4,
          unit="reqps", legend=["{{endpoint}}"]),
    panel(6, "Latency p50 / p95 (/predict)",
          ['histogram_quantile(0.5, sum by (le) (rate('
           'heart_api_request_latency_seconds_bucket{endpoint="/predict"}[5m])))',
           'histogram_quantile(0.95, sum by (le) (rate('
           'heart_api_request_latency_seconds_bucket{endpoint="/predict"}[5m])))'],
          12, 4, unit="s", legend=["p50", "p95"]),
    panel(7, "Predictions by class",
          ["sum by (label) (rate(heart_api_predictions_total[5m]))"], 0, 12,
          unit="ops", legend=["{{label}}"]),
    panel(8, "Predicted probability distribution",
          ["sum by (le) (increase(heart_api_disease_probability_bucket[15m]))"],
          12, 12, kind="bargauge", legend=["<= {{le}}"]),
    panel(9, "Responses by status",
          ["sum by (status) (rate(heart_api_requests_total[1m]))"], 0, 20, 24,
          unit="reqps", legend=["{{status}}"]),
]
panels[7]["targets"][0]["format"] = "heatmap"

dashboard = {
    "uid": "heart-api", "title": "Heart Disease API", "schemaVersion": 39,
    "time": {"from": "now-30m", "to": "now"}, "refresh": "10s",
    "tags": ["mlops", "heart-disease"], "panels": panels,
}
out = Path(__file__).resolve().parent.parent / "monitoring/grafana/dashboards/heart_api.json"
out.write_text(json.dumps(dashboard, indent=2))
print(f"wrote {out}")
