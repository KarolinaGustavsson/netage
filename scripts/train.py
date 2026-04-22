"""Train the CoxMLP and fit the Breslow baseline hazard.

Reads preprocessed train / val CSVs, trains the network with AdamW +
cosine LR schedule + early stopping on val C-index, then fits a sex-stratified
Breslow estimator on the training set predictions of the best checkpoint.

Writes:
  outputs/models/best_model.pt        — best model state dict
  outputs/models/breslow.pkl          — fitted BreslowEstimator
  outputs/models/training_history.json — per-epoch loss and val C-index

Usage::

    python scripts/train.py --config configs/default.yaml
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snakemake / standalone dispatch
# ---------------------------------------------------------------------------

if "snakemake" in dir():
    _config_path = snakemake.config["_config_path"]  # type: ignore[name-defined]
    _in_train = Path(snakemake.input.train)  # type: ignore[name-defined]
    _in_val = Path(snakemake.input.val)  # type: ignore[name-defined]
    _out_ckpt = Path(snakemake.output.checkpoint)  # type: ignore[name-defined]
    _out_breslow = Path(snakemake.output.breslow)  # type: ignore[name-defined]
    _out_history = Path(snakemake.output.history)  # type: ignore[name-defined]
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    _config_path = args.config
    _in_train = _in_val = _out_ckpt = _out_breslow = _out_history = None

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

from amoris_bioage.bioage.breslow import BreslowEstimator
from amoris_bioage.config import load_config
from amoris_bioage.data.schema import FEATURE_COLS
from amoris_bioage.models.network import CoxMLP
from amoris_bioage.training.trainer import Trainer

cfg = load_config(_config_path)
model_dir = Path("outputs/models")
model_dir.mkdir(parents=True, exist_ok=True)

in_train = _in_train or Path(cfg.data.derived_dir) / "train.csv"
in_val = _in_val or Path(cfg.data.derived_dir) / "val.csv"
out_ckpt = _out_ckpt or model_dir / "best_model.pt"
out_breslow = _out_breslow or model_dir / "breslow.pkl"
out_history = _out_history or model_dir / "training_history.json"

train_df = pd.read_csv(in_train)
val_df = pd.read_csv(in_val)

# Feature columns = schema columns + any missing-indicator columns added by
# the preprocessor (named "<col>_missing").
indicator_cols = [c for c in train_df.columns if c.endswith("_missing")]
feature_cols = FEATURE_COLS + indicator_cols
n_features = len(feature_cols)

logger.info(
    "Training on %d individuals (%d features)", len(train_df), n_features
)

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
logger.info("Device: %s", device)

model = CoxMLP(
    n_features=n_features,
    hidden_sizes=cfg.model.hidden_sizes,
    dropout=cfg.model.dropout,
    activation=cfg.model.activation,
)

trainer = Trainer(
    model=model,
    config=cfg.training,
    device=device,
    checkpoint_path=out_ckpt,
)
result = trainer.fit(train_df, val_df, feature_cols)

logger.info(
    "Training complete. Best epoch=%d, val_cindex=%.4f",
    result.best_epoch,
    result.best_val_cindex,
)

with open(out_history, "w") as f:
    json.dump(result.history, f, indent=2)

# Fit Breslow on training set predictions.
logger.info("Fitting Breslow estimator on training set …")
result.model.eval()
ds_features = torch.tensor(train_df[feature_cols].values, dtype=torch.float32).to(device)
ds_age = torch.tensor(train_df["age_at_baseline"].values, dtype=torch.float32).to(device)
with torch.no_grad():
    log_hz = result.model(ds_features, ds_age).cpu().numpy()

breslow = BreslowEstimator()
breslow.fit(
    log_hazard=log_hz,
    event_times=train_df["age_at_exit"].values,
    events=train_df["event"].values,
    entry_times=train_df["age_at_baseline"].values,
    sex=train_df["sex"].values,
)
with open(out_breslow, "wb") as f:
    pickle.dump(breslow, f)

logger.info("Saved model checkpoint to %s", out_ckpt)
logger.info("Saved Breslow estimator to %s", out_breslow)
