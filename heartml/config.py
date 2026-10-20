"""Central configuration: paths, column definitions and data source URLs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = MODELS_DIR / "heart_model.joblib"
MODEL_META_FILE = MODELS_DIR / "model_meta.json"

# Primary mirror is the UCI archive; the second is the UCI static download.
DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data",
    "https://archive.ics.uci.edu/static/public/45/data.csv",
]

# The UCI file has no header row; these are the 14 documented attributes.
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Continuous measurements -> imputed with median and standard-scaled.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Already 0/1 -> imputed with most frequent value, passed through.
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Coded categories without a meaningful order -> one-hot encoded.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "heart-disease-classification"
