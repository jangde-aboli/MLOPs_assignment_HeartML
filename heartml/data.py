"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from heartml import config

log = logging.getLogger(__name__)


def download_raw(dest: Path = config.RAW_FILE, timeout: int = 30) -> Path:
    """Download the raw Cleveland file, trying each mirror in turn."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in config.DATA_URLS:
        try:
            log.info("Downloading %s", url)
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            # The static mirror ships a CSV with a header; normalise it to the
            # header-less, comma-separated layout of processed.cleveland.data.
            if text.lstrip().startswith("age"):
                df = pd.read_csv(io.StringIO(text))
                df.to_csv(dest, header=False, index=False, na_rep="?")
            else:
                dest.write_text(text)
            log.info("Saved raw data to %s", dest)
            return dest
        except Exception as err:  # noqa: BLE001 - try next mirror
            log.warning("Failed to download from %s: %s", url, err)
            last_err = err
    raise RuntimeError(f"Could not download dataset: {last_err}")


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the raw file; '?' marks missing values in the UCI data."""
    return pd.read_csv(path, header=None, names=config.COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame.

    - coerce every column to numeric (bad tokens become NaN)
    - drop exact duplicate rows
    - collapse the 0-4 `num` diagnosis into a binary `target`
    - clip physiologically impossible values to NaN so the pipeline imputes them

    Missing values are deliberately left in place: imputation lives inside the
    model pipeline so that inference applies exactly the same treatment.
    """
    out = df.copy()
    for col in config.COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.drop_duplicates().reset_index(drop=True)

    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])

    out.loc[out["trestbps"] <= 0, "trestbps"] = np.nan
    out.loc[out["chol"] <= 0, "chol"] = np.nan
    return out


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURES], df[config.TARGET]


def prepare(force_download: bool = False) -> pd.DataFrame:
    """Download (if needed), clean and persist the dataset; return it."""
    if force_download or not config.RAW_FILE.exists():
        download_raw()
    df = clean(load_raw())
    config.CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    log.info("Clean dataset: %d rows -> %s", len(df), config.CLEAN_FILE)
    return df
