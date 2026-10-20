import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartml import config
from heartml.data import clean, split_xy
from heartml.features import build_pipeline, build_preprocessor
from heartml.predict import predict_records, risk_band
from heartml.train import candidates, holdout_metrics


@pytest.fixture
def fitted(raw_frame):
    X, y = split_xy(clean(raw_frame))
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    return model, X, y


def test_preprocessor_handles_missing_and_scales(raw_frame):
    X, _ = split_xy(clean(raw_frame))
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()
    n_num = len(config.NUMERIC_FEATURES)
    assert np.allclose(out[:, :n_num].mean(axis=0), 0, atol=1e-7)


def test_preprocessor_one_hot_width(raw_frame):
    X, _ = split_xy(clean(raw_frame))
    pre = build_preprocessor().fit(X)
    names = pre.get_feature_names_out()
    expected = (len(config.NUMERIC_FEATURES) + len(config.BINARY_FEATURES)
                + sum(X[c].nunique() for c in config.CATEGORICAL_FEATURES))
    assert len(names) == expected


def test_preprocessor_ignores_unseen_category(fitted):
    model, X, _ = fitted
    row = X.iloc[[0]].copy()
    row["thal"] = 99
    assert model.predict_proba(row).shape == (1, 2)


def test_pipeline_predicts_probabilities(fitted):
    model, X, _ = fitted
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_metrics_keys_and_ranges(fitted):
    model, X, y = fitted
    m = holdout_metrics(model, X, y)
    assert set(m) == {"test_accuracy", "test_precision", "test_recall",
                      "test_f1", "test_roc_auc"}
    assert all(0 <= v <= 1 for v in m.values())


def test_candidates_include_lr_and_rf():
    names = {c.name for c in candidates()}
    assert {"logistic_regression", "random_forest"} <= names
    for c in candidates(quick=True):
        assert all(len(v) == 1 for v in c.grid.values())


def test_predict_records_output(fitted, sample_patient):
    model, _, _ = fitted
    [r] = predict_records(model, [sample_patient])
    assert r["prediction"] in (0, 1)
    assert 0.5 <= r["confidence"] <= 1
    expected_conf = r["probability_disease"] if r["prediction"] else 1 - r["probability_disease"]
    assert r["confidence"] == pytest.approx(expected_conf, abs=1e-4)
    assert r["risk_level"] in {"low", "moderate", "high"}


def test_predict_records_accepts_missing_optional(fitted, sample_patient):
    model, _, _ = fitted
    patient = {**sample_patient, "ca": None, "thal": None}
    assert len(predict_records(model, [patient])) == 1


@pytest.mark.parametrize("p,band", [(0.1, "low"), (0.45, "moderate"), (0.9, "high")])
def test_risk_band(p, band):
    assert risk_band(p) == band


def test_model_roundtrip(fitted, tmp_path):
    model, X, _ = fitted
    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    pd.testing.assert_frame_equal(
        pd.DataFrame(model.predict_proba(X)), pd.DataFrame(loaded.predict_proba(X)))
