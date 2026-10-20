import numpy as np
import pandas as pd
import pytest

from heartml import config


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Small synthetic frame in the raw UCI layout (with `num`, NaNs, a dup)."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age": rng.integers(30, 75, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
        "num": rng.integers(0, 5, n),
    })
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    df.loc[2, "chol"] = 0  # impossible value
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # duplicate row
    return df[config.COLUMNS]


@pytest.fixture
def sample_patient() -> dict:
    return {"age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 241,
            "fbs": 0, "restecg": 0, "thalach": 123, "exang": 1,
            "oldpeak": 0.2, "slope": 2, "ca": 0, "thal": 7}
