"""Create the EDA / training / inference notebooks (run once, then execute)."""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
SETUP = "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path.cwd().parent))"


def nb(cells):
    book = nbf.v4.new_notebook()
    book.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else
                  nbf.v4.new_code_cell(c) for c in cells]
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                   "language": "python"}
    return book


eda = nb([
    "md:# 01 - Exploratory Data Analysis\nUCI Heart Disease (Cleveland). "
    "Plot code lives in `heartml/eda.py` so the notebook, CI and report use the same figures.",
    SETUP,
    "import pandas as pd\nfrom IPython.display import Image, display\n"
    "from heartml import config\nfrom heartml.data import load_raw, prepare\n"
    "from heartml import eda",
    "md:## Raw data and missing values\nThe raw file uses `?` for missing values.",
    "raw = load_raw() if config.RAW_FILE.exists() else (prepare(), load_raw())[1]\n"
    "raw.head()",
    "raw.isna().sum()[lambda s: s > 0]",
    "md:## Cleaning\n`num` (0-4) becomes binary `target`; duplicates dropped; "
    "impossible 0 values set to NaN. Imputation happens inside the model pipeline.",
    "df = prepare()\ndf.describe().T.round(2)",
    "summary = eda.run()\nsummary['class_counts']",
    "md:## Class balance\nRoughly balanced (54% / 46%), so no resampling is needed; "
    "stratified splits keep the ratio.",
    "display(Image(config.FIGURES_DIR / 'class_balance.png'))",
    "md:## Numeric features",
    "display(Image(config.FIGURES_DIR / 'numeric_histograms.png'))\n"
    "display(Image(config.FIGURES_DIR / 'numeric_boxplots.png'))",
    "md:## Categorical features: disease rate per level",
    "display(Image(config.FIGURES_DIR / 'categorical_disease_rate.png'))",
    "md:## Correlations",
    "display(Image(config.FIGURES_DIR / 'correlation_heatmap.png'))\n"
    "pd.Series(summary['target_correlation'])",
])

train = nb([
    "md:# 02 - Model training & tracking\nRuns the same code as CI "
    "(`python -m heartml.train`). Every candidate is logged to MLflow.",
    SETUP,
    "import logging, json\nimport pandas as pd\nfrom IPython.display import Image, display\n"
    "from heartml import config\nfrom heartml.train import train\n"
    "logging.basicConfig(level=logging.INFO)",
    "meta = train()",
    "pd.read_csv(config.ROOT / 'reports' / 'model_comparison.csv').set_index('model').T",
    "print(json.dumps(meta, indent=2, default=str))",
    "plots = config.ROOT / 'reports' / 'model_plots'\n"
    "for name in ['logistic_regression', 'random_forest', 'gradient_boosting']:\n"
    "    display(Image(plots / f'{name}_roc_curve.png'))",
    "md:Browse all runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db` "
    "from the project root.",
])

infer = nb([
    "md:# 03 - Inference\nLoad the saved pipeline and score new patients "
    "(raw feature values; preprocessing is inside the pipeline).",
    SETUP,
    "import json\nfrom heartml import config\nfrom heartml.predict import load_model, "
    "predict_records",
    "model = load_model()\nmodel",
    "patients = json.loads((config.ROOT / 'samples' / 'batch.json').read_text())['patients']\n"
    "predict_records(model, patients)",
    "md:Missing `ca`/`thal` are allowed; the pipeline imputes them.",
    "predict_records(model, [{**patients[0], 'ca': None, 'thal': None}])",
])

for name, book in [("01_eda", eda), ("02_training", train), ("03_inference", infer)]:
    nbf.write(book, NB_DIR / f"{name}.ipynb")
    print("wrote", name)
