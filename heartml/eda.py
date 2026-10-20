"""Exploratory data analysis: writes figures and a summary to reports/."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heartml import config  # noqa: E402
from heartml.data import load_raw, prepare  # noqa: E402

log = logging.getLogger(__name__)

PALETTE = {0: "#4C78A8", 1: "#E45756"}
LABELS = {0: "No disease", 1: "Disease"}


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 2,
                f"{n} ({n / counts.sum():.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "class_balance.png", out_dir)


def plot_numeric_histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.6))
    for ax, col in zip(axes, cols):
        for t in (0, 1):
            ax.hist(df.loc[df[config.TARGET] == t, col].dropna(), bins=20,
                    alpha=0.6, color=PALETTE[t], label=LABELS[t])
        ax.set_title(col)
    axes[0].legend()
    fig.suptitle("Numeric feature distributions by diagnosis")
    return _save(fig, "numeric_histograms.png", out_dir)


def plot_categorical_rates(df: pd.DataFrame, out_dir: Path) -> Path:
    """Disease rate per category level - more informative than raw counts."""
    cols = config.BINARY_FEATURES + config.CATEGORICAL_FEATURES
    ncols = 4
    nrows = -(-len(cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.3 * nrows))
    for ax, col in zip(axes.flat, cols):
        rate = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        ax.bar(rate.index.astype(int).astype(str), rate["mean"], color="#72B7B2")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{col}: disease rate")
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    return _save(fig, "categorical_disease_rate.png", out_dir)


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = df[config.FEATURES + [config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Correlation matrix (Pearson)")
    return _save(fig, "correlation_heatmap.png", out_dir)


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    long = df.melt(id_vars=config.TARGET, value_vars=config.NUMERIC_FEATURES)
    long[config.TARGET] = long[config.TARGET].map(LABELS)
    g = sns.catplot(data=long, x=config.TARGET, y="value", col="variable",
                    kind="box", sharey=False, height=3.4, aspect=0.8,
                    hue=config.TARGET,
                    palette={LABELS[k]: v for k, v in PALETTE.items()})
    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    path = out_dir / "numeric_boxplots.png"
    g.savefig(path, dpi=130)
    plt.close(g.figure)
    return path


def summarise(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "missing_in_raw": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "class_counts": {LABELS[int(k)]: int(v)
                         for k, v in df[config.TARGET].value_counts().items()},
        "target_correlation": df[config.FEATURES + [config.TARGET]].corr()[
            config.TARGET].drop(config.TARGET).round(3)
        .sort_values(key=abs, ascending=False).to_dict(),
    }


def run(out_dir: Path = config.FIGURES_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare()
    raw = load_raw()
    sns.set_theme(style="whitegrid")
    figures = [
        plot_class_balance(df, out_dir),
        plot_numeric_histograms(df, out_dir),
        plot_categorical_rates(df, out_dir),
        plot_correlation(df, out_dir),
        plot_boxplots(df, out_dir),
    ]
    summary = summarise(raw, df)
    summary["figures"] = [p.name for p in figures]
    (out_dir.parent / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("EDA figures written to %s", out_dir)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    print(json.dumps(run(), indent=2))
