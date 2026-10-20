"""Hit a running API with sample requests; optionally generate load.

    python scripts/smoke_test.py --url http://localhost:8000
    python scripts/smoke_test.py --url http://127.0.0.1 --load 300   # fills Grafana

Exits non-zero if any check fails, so CI can use it after `docker run`.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def wait_ready(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=2).json().get("model_loaded"):
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    sys.exit(f"API at {url} not ready after {timeout}s")


def random_patient(rng: random.Random) -> dict:
    return {"age": rng.randint(29, 77), "sex": rng.randint(0, 1),
            "cp": rng.randint(1, 4), "trestbps": rng.randint(94, 200),
            "chol": rng.randint(126, 400), "fbs": rng.randint(0, 1),
            "restecg": rng.randint(0, 2), "thalach": rng.randint(71, 202),
            "exang": rng.randint(0, 1), "oldpeak": round(rng.uniform(0, 5), 1),
            "slope": rng.randint(1, 3), "ca": rng.randint(0, 3),
            "thal": rng.choice([3, 6, 7])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--load", type=int, default=0, help="extra random requests")
    ap.add_argument("--wait", type=float, default=60)
    args = ap.parse_args()
    url = args.url.rstrip("/")

    wait_ready(url, args.wait)
    patient = json.loads((SAMPLES / "patient.json").read_text())

    r = requests.post(f"{url}/predict", json=patient, timeout=5)
    r.raise_for_status()
    body = r.json()
    print("POST /predict ->", json.dumps(body))
    assert body["prediction"] in (0, 1) and 0.5 <= body["confidence"] <= 1

    r = requests.post(f"{url}/predict/batch",
                      json=json.loads((SAMPLES / "batch.json").read_text()), timeout=5)
    r.raise_for_status()
    print("POST /predict/batch ->", len(r.json()["predictions"]), "predictions")

    bad = requests.post(f"{url}/predict", json={**patient, "cp": 9}, timeout=5)
    assert bad.status_code == 422, bad.status_code
    print("invalid input correctly rejected with 422")

    assert "heart_api_requests_total" in requests.get(f"{url}/metrics", timeout=5).text
    print("/metrics exposes Prometheus counters")

    rng = random.Random(7)
    for i in range(args.load):
        requests.post(f"{url}/predict", json=random_patient(rng), timeout=5)
        if i % 10 == 9:  # sprinkle in some invalid requests for the error panel
            requests.post(f"{url}/predict", json={"age": -1}, timeout=5)
        time.sleep(0.05)
    if args.load:
        print(f"sent {args.load} load-test requests")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
