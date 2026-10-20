"""Preprocessing pipeline shared by training and serving."""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartml import config


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Unseen codes at inference time map to all-zeros instead of failing.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("bin", binary, config.BINARY_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Preprocessor + estimator as one object, so the saved artefact is
    self-contained and takes raw feature values."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
