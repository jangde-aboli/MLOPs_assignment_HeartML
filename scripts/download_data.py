"""Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV.

    python scripts/download_data.py            # raw -> data/raw, clean -> data/processed
    python scripts/download_data.py --force    # re-download even if present
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heartml.data import prepare  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare(force_download=ap.parse_args().force)
    print(df.head())
    print(f"\n{len(df)} rows, target counts: {df['target'].value_counts().to_dict()}")
