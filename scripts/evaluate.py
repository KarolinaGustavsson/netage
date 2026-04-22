"""Compute validation and test metrics for the trained CoxMLP.

Evaluates:
  - Harrell C-index on val and test splits
  - Calibration table (predicted vs observed 10-year mortality) on test
  - Incremental C-index LRT: Cox(Δ + age) vs Cox(age) on test

Writes outputs/results/metrics.json.

Usage::

    python scripts/evaluate.py --config configs/default.yaml
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
    _in_test = Path(snakemake.input.test)  # type: ignore[name-defined]
    _in_bioage = Path(snakemake.input.bioage)  # type: ignore[name-defined]
    _in_ckpt = Path(snakemake.input.checkpoint)  # type: ignore[name-defined]
    _in_breslow = Path(snakemake.input.breslow)  # type: ignore[name-defined]
    _out_metrics = Path(snakemake.output[0])  # type: ignore[name-defined]
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    _config_path = args.config
    _in_test = _in_bioage = _in_ckpt = _in_breslow = _out_metrics = None

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

from amoris_bioage.config import load_config
from amoris_bioage.data.schema import FEATURE_COLS
from amoris_bioage.models.network import CoxMLP
from amoris_bioage.validation.calibration import calibration_by_decile
from amoris_bioage.validation.concordance import compute_cindex
from amoris_bioage.validation.incremental import incremental_cindex_lrt

cfg = load_config(_config_path)
results_dir = Path("outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path("outputs/models")

in_test = _in_test or Path(cfg.data.derived_dir) / "test.csv"
in_bioage = _in_bioage or results_dir / "bioage_test.csv"
in_ckpt = _in_ckpt or model_dir / "best_model.pt"
in_breslow = _in_breslow or model_dir / "breslow.pkl"
out_metrics = _out_metrics or results_dir / "metrics.json"

test_df = pd.read_csv(in_test)
bioage_df = pd.read_csv(in_bioage)

indicator_cols = [c for c in test_df.columns if c.endswith("_missing")]
feature_cols = FEATURE_COLS + indicator_cols
n_features = len(feature_cols)

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CoxMLP(
    n_features=n_features,
    hidden_sizes=cfg.model.hidden_sizes,
    dropout=cfg.model.dropout,
    activation=cfg.model.activation,
)
state = torch.load(in_ckpt, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval().to(device)

features_t = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(device)
age_t = torch.tensor(test_df["age_at_baseline"].values, dtype=torch.float32).to(device)
with torch.no_grad():
    log_hz = model(features_t, age_t).cpu().numpy()

# C-index on test set.
cindex_test = compute_cindex(log_hz, test_df["age_at_exit"].values, test_df["event"].values)
logger.info("Test C-index: %.4f", cindex_test)

# Calibration by decile.
with open(in_breslow, "rb") as f:
    breslow = pickle.load(f)

follow_up = test_df["age_at_exit"].values - test_df["age_at_baseline"].values
from amoris_bioage.bioage.breslow import BreslowEstimator
h0_t = breslow.predict_cumhaz(test_df["age_at_baseline"].values, test_df["sex"].values)
h0_t_h = breslow.predict_cumhaz(
    test_df["age_at_baseline"].values + cfg.bioage.horizon_years,
    test_df["sex"].values,
)
pred_mortality = 1.0 - np.exp(-np.exp(log_hz) * (h0_t_h - h0_t))

calib_df = calibration_by_decile(pred_mortality, follow_up, test_df["event"].values)

# Incremental C-index LRT: Δ vs age alone.
delta = bioage_df["delta"].values
lrt_result = incremental_cindex_lrt(
    delta=delta,
    age=test_df["age_at_baseline"].values,
    event_times=test_df["age_at_exit"].values,
    events=test_df["event"].values,
    entry_times=test_df["age_at_baseline"].values,
)
logger.info(
    "Incremental C-index: %.4f → %.4f (Δ=%.4f), LRT p=%.4e",
    lrt_result["c_null"],
    lrt_result["c_full"],
    lrt_result["delta_c"],
    lrt_result["p_value"],
)

metrics = {
    "cindex_test": float(cindex_test),
    "calibration": calib_df.to_dict(orient="records"),
    "incremental_lrt": {k: float(v) for k, v in lrt_result.items()},
}
with open(out_metrics, "w") as f:
    json.dump(metrics, f, indent=2)

logger.info("Metrics saved to %s", out_metrics)
