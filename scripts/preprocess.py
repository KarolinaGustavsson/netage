"""Preprocess raw AMORIS data into train / validation / test splits.

Reads the raw CSV, applies quality control, splits by sex × age-decile
stratum, fits the Preprocessor on the training set, and writes:

  outputs/derived/train.csv
  outputs/derived/val.csv
  outputs/derived/test.csv
  outputs/derived/preprocessor.pkl

Can be run directly::

    python scripts/preprocess.py --config configs/default.yaml

or invoked by Snakemake (snakemake object is available in that context).
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snakemake / standalone dispatch
# ---------------------------------------------------------------------------

if "snakemake" in dir():  # running inside Snakemake
    _config_path = snakemake.config["_config_path"]  # type: ignore[name-defined]
    _out_train = Path(snakemake.output.train)  # type: ignore[name-defined]
    _out_val = Path(snakemake.output.val)  # type: ignore[name-defined]
    _out_test = Path(snakemake.output.test)  # type: ignore[name-defined]
    _out_prep = Path(snakemake.output.preprocessor)  # type: ignore[name-defined]
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    _config_path = args.config
    _out_train = None
    _out_val = None
    _out_test = None
    _out_prep = None

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

from amoris_bioage.config import load_config
from amoris_bioage.data.loader import load_raw
from amoris_bioage.data.preprocessing import Preprocessor
from amoris_bioage.data.schema import FEATURE_COLS
from amoris_bioage.data.splits import make_splits

cfg = load_config(_config_path)

out_dir = Path(cfg.data.derived_dir)
out_dir.mkdir(parents=True, exist_ok=True)

out_train = _out_train or out_dir / "train.csv"
out_val = _out_val or out_dir / "val.csv"
out_test = _out_test or out_dir / "test.csv"
out_prep = _out_prep or out_dir / "preprocessor.pkl"

logger.info("Loading raw data from %s", cfg.data.raw_path)
df = load_raw(cfg.data.raw_path)
logger.info("Loaded %d individuals", len(df))

splits = make_splits(
    df,
    ratios=(cfg.data.train_frac, cfg.data.val_frac, cfg.data.test_frac),
    seed=cfg.data.split_seed,
)
logger.info(
    "Splits: train=%d val=%d test=%d",
    len(splits.train),
    len(splits.val),
    len(splits.test),
)

preprocessor = Preprocessor()
train_pp = preprocessor.fit_transform(splits.train, FEATURE_COLS)
val_pp = preprocessor.transform(splits.val)
test_pp = preprocessor.transform(splits.test)

train_pp.to_csv(out_train, index=False)
val_pp.to_csv(out_val, index=False)
test_pp.to_csv(out_test, index=False)
with open(out_prep, "wb") as f:
    pickle.dump(preprocessor, f)

logger.info("Wrote train/val/test to %s", out_dir)
