import numpy as np

from heartml import config
from heartml.data import clean, load_raw, split_xy


def test_clean_creates_binary_target(raw_frame):
    out = clean(raw_frame)
    assert config.TARGET in out.columns
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    # num > 0 means disease
    expected = (raw_frame.drop_duplicates()["num"] > 0).astype(int).to_numpy()
    assert np.array_equal(out[config.TARGET].to_numpy(), expected)


def test_clean_drops_duplicates(raw_frame):
    out = clean(raw_frame)
    assert len(out) == len(raw_frame) - 1
    assert not out.duplicated().any()


def test_clean_marks_impossible_values_missing(raw_frame):
    out = clean(raw_frame)
    assert out["chol"].isna().sum() == 1
    assert (out["chol"].dropna() > 0).all()


def test_clean_keeps_missing_for_pipeline(raw_frame):
    out = clean(raw_frame)
    assert out["ca"].isna().sum() == 1
    assert out["thal"].isna().sum() == 1


def test_clean_does_not_mutate_input(raw_frame):
    before = raw_frame.copy()
    clean(raw_frame)
    assert raw_frame.equals(before)


def test_load_raw_parses_question_marks(tmp_path):
    f = tmp_path / "raw.data"
    f.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n"
                 "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,?,2\n")
    df = load_raw(f)
    assert list(df.columns) == config.COLUMNS
    assert df["ca"].isna().sum() == 1 and df["thal"].isna().sum() == 1


def test_split_xy_columns(raw_frame):
    X, y = split_xy(clean(raw_frame))
    assert list(X.columns) == config.FEATURES
    assert len(X) == len(y)
