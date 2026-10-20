"""Inference helpers used by the API and the CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from heartml import config


def load_model(path: Path = config.MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run `python -m heartml.train` first.")
    return joblib.load(path)


def risk_band(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "moderate"
    return "high"


def predict_records(model, records: list[dict], threshold: float = 0.5) -> list[dict]:
    frame = pd.DataFrame(records, columns=config.FEATURES)
    proba = model.predict_proba(frame)[:, 1]
    results = []
    for p in proba:
        label = int(p >= threshold)
        results.append({
            "prediction": label,
            "label": "heart disease" if label else "no heart disease",
            "probability_disease": round(float(p), 4),
            # Confidence = probability of the predicted class.
            "confidence": round(float(p if label else 1 - p), 4),
            "risk_level": risk_band(float(p)),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict from a JSON file")
    parser.add_argument("json_file", help="file with one record or a list")
    args = parser.parse_args()
    payload = json.loads(Path(args.json_file).read_text())
    records = payload if isinstance(payload, list) else [payload]
    print(json.dumps(predict_records(load_model(), records), indent=2))


if __name__ == "__main__":
    main()
