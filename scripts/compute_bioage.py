"""Compute biological age g and age gap Δ for train / val / test splits.

Loads the trained CoxMLP and Breslow estimator, fits the reference mortality
mapping (BiologicalAgeEstimator), and writes a CSV per split with columns:

  id, sex, age_at_baseline, age_at_exit, event, g, delta

Usage::

    python scripts/compute_bioage.py --config configs/default.yaml
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snakemake / standalone dispatch
# ---------------------------------------------------------------------------

if "snakemake" in dir():
    _config_path = snakemake.config["_config_path"]  # type: ignore[name-defined]
    _in_ckpt = Path(snakemake.input.checkpoint)  # type: ignore[name-defined]
    _in_breslow = Path(snakemake.input.breslow)  # type: ignore[name-defined]
    _out_test = Path(snakemake.output.test)  # type: ignore[name-defined]
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    _config_path = args.config
    _in_ckpt = _in_breslow = _out_test = None

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

from amoris_bioage.bioage.inversion import BiologicalAgeEstimator, sanity_check
from amoris_bioage.config import load_config
from amoris_bioage.data.schema import FEATURE_COLS
from amoris_bioage.models.network import CoxMLP

cfg = load_config(_config_path)
model_dir = Path("outputs/models")
results_dir = Path("outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)

in_ckpt = _in_ckpt or model_dir / "best_model.pt"
in_breslow = _in_breslow or model_dir / "breslow.pkl"

with open(in_breslow, "rb") as f:
    breslow = pickle.load(f)

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Determine feature_cols from one of the split files.
_sample_df = pd.read_csv(Path(cfg.data.derived_dir) / "train.csv", nrows=1)
indicator_cols = [c for c in _sample_df.columns if c.endswith("_missing")]
feature_cols = FEATURE_COLS + indicator_cols
n_features = len(feature_cols)

model = CoxMLP(
    n_features=n_features,
    hidden_sizes=cfg.model.hidden_sizes,
    dropout=cfg.model.dropout,
    activation=cfg.model.activation,
)
state = torch.load(in_ckpt, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval().to(device)

bioage_est = BiologicalAgeEstimator(
    horizon_years=cfg.bioage.horizon_years,
    age_grid_min=cfg.bioage.age_grid_min,
    age_grid_max=cfg.bioage.age_grid_max,
    age_grid_step=cfg.bioage.age_grid_step,
)
bioage_est.fit_reference(breslow)


def _zscore_finite(x: pd.Series) -> np.ndarray:
    """Z-score finite values; keep non-finite entries as NaN."""
    arr = x.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    vals = arr[mask]
    std = vals.std()
    if std <= 0:
        return out
    out[mask] = (vals - vals.mean()) / std
    return out


def _compute_bioage_for_split(split_path: Path) -> pd.DataFrame:
    df = pd.read_csv(split_path)
    features_t = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
    age_t = torch.tensor(df["age_at_baseline"].values, dtype=torch.float32).to(device)
    with torch.no_grad():
        log_hz = model(features_t, age_t).cpu().numpy()
    g, delta = bioage_est.transform(
        log_hz, df["age_at_baseline"].values, df["sex"].values, breslow
    )
    result = df[["id", "sex", "age_at_baseline", "age_at_exit", "event"]].copy()
    result["g"] = g
    result["delta"] = delta
    result["delta_sd"] = _zscore_finite(result["delta"])
    return result


derived = Path(cfg.data.derived_dir)
for split, out_path in [
    ("train", results_dir / "bioage_train.csv"),
    ("val", results_dir / "bioage_val.csv"),
    ("test", _out_test or results_dir / "bioage_test.csv"),
]:
    split_df = _compute_bioage_for_split(derived / f"{split}.csv")
    sc = sanity_check(split_df["delta"].values, split_df["age_at_baseline"].values)
    logger.info(
        "%s: mean_Δ=%.3f std_Δ=%.3f corr(Δ,age)=%.3f mean_check=%s",
        split,
        sc["mean_delta"],
        sc["std_delta"],
        sc["corr_delta_age"],
        sc["mean_check_passed"],
    )
    split_df.to_csv(out_path, index=False)
    logger.info("Saved %s bioage to %s", split, out_path)
