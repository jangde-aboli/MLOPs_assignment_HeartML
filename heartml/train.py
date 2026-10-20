"""Train, tune and compare classifiers; track every run in MLflow.

Each candidate model is tuned with GridSearchCV (stratified 5-fold, ROC-AUC)
on the training split, then scored on a held-out test split. All runs are
logged to MLflow. The model with the best cross-validated ROC-AUC is saved
as a joblib pipeline (used by the API) and as an MLflow model.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from heartml import config  # noqa: E402
from heartml.data import prepare, split_xy  # noqa: E402
from heartml.features import build_pipeline  # noqa: E402

log = logging.getLogger(__name__)

# Types skops refuses by default. Safe here: we only load models we trained.
# numpy.dtype comes from OneHotEncoder, Tree from the tree ensembles.
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.tree._tree.Tree"]

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


@dataclass
class Candidate:
    name: str
    estimator: object
    grid: dict = field(default_factory=dict)


def candidates(quick: bool = False) -> list[Candidate]:
    """Model search space. `quick` shrinks the grids for CI smoke runs."""
    rs = config.RANDOM_STATE
    lr_grid = {"model__C": [0.01, 0.1, 1.0, 10.0],
               "model__penalty": ["l1", "l2"]}
    rf_grid = {"model__n_estimators": [200, 400],
               "model__max_depth": [None, 4, 8],
               "model__min_samples_leaf": [1, 3, 5],
               "model__max_features": ["sqrt", 0.5]}
    gb_grid = {"model__n_estimators": [100, 200],
               "model__learning_rate": [0.03, 0.1],
               "model__max_depth": [2, 3],
               "model__subsample": [0.8, 1.0]}
    if quick:
        lr_grid = {"model__C": [1.0], "model__penalty": ["l2"]}
        rf_grid = {"model__n_estimators": [100], "model__max_depth": [4]}
        gb_grid = {"model__n_estimators": [50], "model__max_depth": [2]}
    return [
        Candidate("logistic_regression",
                  LogisticRegression(solver="liblinear", max_iter=2000,
                                     random_state=rs), lr_grid),
        Candidate("random_forest",
                  RandomForestClassifier(random_state=rs, n_jobs=-1), rf_grid),
        Candidate("gradient_boosting",
                  GradientBoostingClassifier(random_state=rs), gb_grid),
    ]


def holdout_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred),
        "test_f1": f1_score(y, pred),
        "test_roc_auc": roc_auc_score(y, proba),
    }


def _plot_artifacts(model, X_test, y_test, name: str, out_dir: Path) -> list[Path]:
    paths = []
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax)
    ax.set_title(f"{name}: confusion matrix")
    fig.tight_layout()
    paths.append(out_dir / f"{name}_confusion_matrix.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"{name}: ROC curve")
    fig.tight_layout()
    paths.append(out_dir / f"{name}_roc_curve.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    importances = feature_importance(model)
    if importances is not None:
        top = importances.head(15)[::-1]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.barh(top.index, top.values, color="#4C78A8")
        ax.set_title(f"{name}: top features")
        fig.tight_layout()
        paths.append(out_dir / f"{name}_feature_importance.png")
        fig.savefig(paths[-1], dpi=120)
        plt.close(fig)
    return paths


def feature_importance(model) -> pd.Series | None:
    est = model.named_steps["model"]
    names = model.named_steps["preprocess"].get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_[0])
    else:
        return None
    return pd.Series(values, index=names).sort_values(ascending=False)


def train(quick: bool = False, test_size: float = 0.2) -> dict:
    df = prepare()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    if os.getenv("MLFLOW_TRACKING_URI") is None:
        mlflow.set_tracking_uri(f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    plot_dir = config.ROOT / "reports" / "model_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cand in candidates(quick):
        log.info("Tuning %s over %d combinations", cand.name,
                 int(np.prod([len(v) for v in cand.grid.values()])))
        with mlflow.start_run(run_name=cand.name) as run:
            search = GridSearchCV(build_pipeline(cand.estimator), cand.grid,
                                  scoring="roc_auc", cv=cv, n_jobs=-1,
                                  refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            # Every metric from 5-fold CV of the tuned configuration.
            cv_scores = cross_validate(best, X_train, y_train, cv=cv,
                                       scoring=SCORING, n_jobs=-1)
            metrics = {f"cv_{m}_mean": float(cv_scores[f"test_{m}"].mean())
                       for m in SCORING}
            metrics.update({f"cv_{m}_std": float(cv_scores[f"test_{m}"].std())
                            for m in SCORING})
            metrics.update(holdout_metrics(best, X_test, y_test))

            mlflow.set_tags({"model_family": cand.name, "quick_mode": str(quick)})
            mlflow.log_params({k.replace("model__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_params({"test_size": test_size, "cv_folds": 5,
                               "n_train": len(X_train), "n_test": len(X_test),
                               "grid_size": len(search.cv_results_["params"])})
            mlflow.log_metrics(metrics)

            for p in _plot_artifacts(best, X_test, y_test, cand.name, plot_dir):
                mlflow.log_artifact(str(p), artifact_path="plots")
            grid_csv = plot_dir / f"{cand.name}_grid_results.csv"
            pd.DataFrame(search.cv_results_).to_csv(grid_csv, index=False)
            mlflow.log_artifact(str(grid_csv), artifact_path="tuning")

            mlflow.sklearn.log_model(
                best, name="model",
                signature=infer_signature(X_train, best.predict_proba(X_train)),
                input_example=X_train.head(3), skops_trusted_types=SKOPS_TRUSTED)

            log.info("%s: cv_roc_auc=%.3f test_roc_auc=%.3f",
                     cand.name, metrics["cv_roc_auc_mean"], metrics["test_roc_auc"])
            results.append({"name": cand.name, "run_id": run.info.run_id,
                            "best_params": search.best_params_,
                            "metrics": metrics, "model": best})

    winner = max(results, key=lambda r: r["metrics"]["cv_roc_auc_mean"])
    log.info("Selected %s (cv ROC-AUC %.3f)", winner["name"],
             winner["metrics"]["cv_roc_auc_mean"])

    # Final model is refit on all data with the winning hyperparameters.
    final = build_pipeline(winner["model"].named_steps["model"])
    final.fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_FILE)
    meta = {
        "model_name": winner["name"],
        "mlflow_run_id": winner["run_id"],
        "best_params": winner["best_params"],
        "metrics": {k: round(v, 4) for k, v in winner["metrics"].items()},
        "features": config.FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_samples": len(X),
        "quick_mode": quick,
    }
    config.MODEL_META_FILE.write_text(json.dumps(meta, indent=2, default=str))

    with mlflow.start_run(run_name=f"final_{winner['name']}"):
        mlflow.set_tags({"stage": "final", "selected_from": winner["run_id"]})
        mlflow.log_params({k.replace("model__", ""): v
                           for k, v in winner["best_params"].items()})
        mlflow.log_artifact(str(config.MODEL_FILE), artifact_path="joblib")
        mlflow.log_artifact(str(config.MODEL_META_FILE))
        mlflow.sklearn.log_model(
            final, name="model", registered_model_name="heart-disease-classifier",
            signature=infer_signature(X, final.predict_proba(X)),
            input_example=X.head(3), skops_trusted_types=SKOPS_TRUSTED)

    comparison = pd.DataFrame(
        [{"model": r["name"], **r["metrics"]} for r in results]).round(4)
    comparison_path = config.ROOT / "reports" / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(comparison[["model", "cv_roc_auc_mean", "cv_accuracy_mean",
                      "test_accuracy", "test_precision", "test_recall",
                      "test_roc_auc"]].to_string(index=False))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="tiny hyperparameter grids (for CI)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = train(quick=args.quick)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
