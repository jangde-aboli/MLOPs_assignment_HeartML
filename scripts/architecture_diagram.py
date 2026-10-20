"""Render reports/figures/architecture.png for the report."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports/figures/architecture.png"

GROUPS = [  # (x, y, w, h, title, colour)
    (0.2, 4.6, 5.6, 2.2, "Development / training", "#E8F1FA"),
    (6.2, 4.6, 5.6, 2.2, "CI/CD - GitHub Actions", "#FDF0E6"),
    (0.2, 0.3, 11.6, 3.7, "Kubernetes cluster - namespace heart-mlops", "#EAF6EC"),
]
BOXES = {  # name: (x, y, label)
    "uci": (0.5, 5.1, "UCI repo\n(download_data.py)"),
    "train": (2.3, 5.1, "EDA + train.py\nLR / RF / GB\nGridSearchCV"),
    "mlflow": (4.1, 5.1, "MLflow\nruns, metrics,\nplots, registry"),
    "lint": (6.45, 5.1, "flake8"),
    "test": (7.75, 5.1, "pytest"),
    "ci_train": (9.05, 5.1, "train +\nquality gate"),
    "docker": (10.35, 5.1, "docker build\n+ smoke test"),
    "client": (0.5, 1.6, "Client\n(curl / app)"),
    "lb": (2.4, 1.6, "LoadBalancer\nService / Ingress"),
    "pods": (4.6, 1.6, "heart-api pods x2\nFastAPI + model\n/predict /metrics"),
    "prom": (7.2, 1.6, "Prometheus\n(pod discovery)"),
    "graf": (9.6, 1.6, "Grafana\ndashboard"),
    "logs": (4.6, 0.5, "JSON request logs\n(kubectl logs)"),
}
W, H = 1.5, 1.1
CI = ("lint", "test", "ci_train", "docker")
ARROWS = [("uci", "train"), ("train", "mlflow"), ("lint", "test"), ("test", "ci_train"),
          ("ci_train", "docker"), ("client", "lb"), ("lb", "pods"), ("prom", "pods"),
          ("prom", "graf"), ("pods", "logs")]

fig, ax = plt.subplots(figsize=(13, 7.4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")
for x, y, w, h, title, colour in GROUPS:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=colour, ec="#999999", lw=1))
    ax.text(x + 0.15, y + h - 0.25, title, fontsize=11, weight="bold", va="top")
for name, (x, y, label) in BOXES.items():
    h = 0.7 if name == "logs" else H
    w = 2.0 if name in ("pods", "logs") else 1.05 if name in CI else W
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                fc="white", ec="#4C78A8", lw=1.4))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    BOXES[name] = (x, y, w, h)


def anchor(name, side):
    x, y, w, h = BOXES[name]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


for a, b in ARROWS:
    if b == "logs":
        start, end = anchor(a, "b"), anchor(b, "t")
    elif a == "prom" and b == "pods":
        start, end = anchor(a, "l"), anchor(b, "r")
    else:
        start, end = anchor(a, "r"), anchor(b, "l")
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 color="#333333", lw=1.2))

# cross-group flows
ax.add_patch(FancyArrowPatch(anchor("mlflow", "r"), anchor("lint", "l"),
                             arrowstyle="-|>", mutation_scale=14, ls="--", color="#777"))
ax.text(5.95, 5.85, "git push", ha="center", fontsize=8, color="#555")
ax.add_patch(FancyArrowPatch(anchor("docker", "b"), (5.6, 2.7), arrowstyle="-|>",
                             mutation_scale=14, ls="--", color="#777",
                             connectionstyle="arc3,rad=-0.15"))
ax.text(8.6, 3.75, "image heart-api:1.0.0  (kubectl apply -f k8s/)", fontsize=8, color="#555")
ax.text(6.85, 2.35, "scrape", fontsize=8, color="#555")
ax.set_title("Heart disease MLOps architecture", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
