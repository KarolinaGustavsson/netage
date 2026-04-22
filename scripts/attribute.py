"""Compute SHAP values for g and Δ on the test set.

Builds the age-stratified background from the training set, then runs
BioageShapExplainer on the test set to produce:

  outputs/results/shap_g.npz       — SHAP values for biological age g
  outputs/results/shap_delta.npz   — SHAP values for age gap Δ
  outputs/results/ig_eta.npz       — Integrated gradients for η (optional)

Each .npz file contains:
  - values:      (N, n_input_cols) SHAP values
  - base_values: (N,) expected output over background
  - feature_names: string array of input column names

SHAP computation is model-agnostic (permutation SHAP); it scales linearly
with n_test * n_background * n_features.  For very large test sets, use
--n-explain to limit the number of test individuals explained.

Usage::

    python scripts/attribute.py --config configs/default.yaml [--n-explain 500]
"""
from __future__ import annotations

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
    _in_test = Path(snakemake.input.test)  # type: ignore[name-defined]
    _in_ckpt = Path(snakemake.input.checkpoint)  # type: ignore[name-defined]
    _in_breslow = Path(snakemake.input.breslow)  # type: ignore[name-defined]
    _in_bioage = Path(snakemake.input.bioage)  # type: ignore[name-defined]
    _out_shap_g = Path(snakemake.output.shap_g)  # type: ignore[name-defined]
    _out_shap_delta = Path(snakemake.output.shap_delta)  # type: ignore[name-defined]
    _n_explain = snakemake.params.get("n_explain", 500)  # type: ignore[name-defined]
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-explain", type=int, default=500,
                        help="Number of test individuals to explain (default 500)")
    args = parser.parse_args()
    _config_path = args.config
    _n_explain = args.n_explain
    _in_train = _in_test = _in_ckpt = _in_breslow = _in_bioage = None
    _out_shap_g = _out_shap_delta = None

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

from amoris_bioage.attribution.background import make_age_stratified_background
from amoris_bioage.attribution.ig_explainer import CoxIGExplainer
from amoris_bioage.attribution.shap_explainer import BioageShapExplainer
from amoris_bioage.bioage.inversion import BiologicalAgeEstimator
from amoris_bioage.config import load_config
from amoris_bioage.data.schema import FEATURE_COLS
from amoris_bioage.models.network import CoxMLP

cfg = load_config(_config_path)
model_dir = Path("outputs/models")
results_dir = Path("outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)

in_train = _in_train or Path(cfg.data.derived_dir) / "train.csv"
in_test = _in_test or Path(cfg.data.derived_dir) / "test.csv"
in_ckpt = _in_ckpt or model_dir / "best_model.pt"
in_breslow = _in_breslow or model_dir / "breslow.pkl"
out_shap_g = _out_shap_g or results_dir / "shap_g.npz"
out_shap_delta = _out_shap_delta or results_dir / "shap_delta.npz"

train_df = pd.read_csv(in_train)
test_df = pd.read_csv(in_test)

indicator_cols = [c for c in train_df.columns if c.endswith("_missing")]
feature_cols = FEATURE_COLS + indicator_cols
n_features = len(feature_cols)

device = "cpu"  # SHAP wrapper uses numpy; keep model on CPU for thread safety

with open(in_breslow, "rb") as f:
    breslow = pickle.load(f)

model = CoxMLP(
    n_features=n_features,
    hidden_sizes=cfg.model.hidden_sizes,
    dropout=cfg.model.dropout,
    activation=cfg.model.activation,
)
state = torch.load(in_ckpt, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

bioage_est = BiologicalAgeEstimator(
    horizon_years=cfg.bioage.horizon_years,
    age_grid_min=cfg.bioage.age_grid_min,
    age_grid_max=cfg.bioage.age_grid_max,
    age_grid_step=cfg.bioage.age_grid_step,
)
bioage_est.fit_reference(breslow)

background = make_age_stratified_background(train_df, feature_cols, n_age_bins=10)
logger.info("Background: %d rows × %d features", len(background), len(feature_cols))

explainer = BioageShapExplainer(
    model=model,
    breslow=breslow,
    bioage_estimator=bioage_est,
    feature_cols=feature_cols,
    background=background,
    device=device,
)

# Explain a subset of the test set.
test_subset = test_df.sample(
    n=min(_n_explain, len(test_df)), random_state=0
).reset_index(drop=True)
logger.info("Explaining %d individuals …", len(test_subset))

input_cols = feature_cols + ["age_at_baseline", "sex"]

logger.info("Computing SHAP values for g …")
expl_g = explainer.explain_g(test_subset)
np.savez(
    out_shap_g,
    values=expl_g.values,
    base_values=expl_g.base_values,
    feature_names=np.array(input_cols),
)

logger.info("Computing SHAP values for Δ …")
expl_delta = explainer.explain_delta(test_subset)
np.savez(
    out_shap_delta,
    values=expl_delta.values,
    base_values=expl_delta.base_values,
    feature_names=np.array(input_cols),
)

# Integrated gradients on η (fast, differentiable).
logger.info("Computing integrated gradients for η …")
ig_explainer = CoxIGExplainer(model, feature_cols, background)
ig_result = ig_explainer.explain(test_subset, n_steps=50)
np.savez(
    results_dir / "ig_eta.npz",
    feature_attributions=ig_result["feature_attributions"],
    age_attribution=ig_result["age_attribution"],
    eta=ig_result["eta"],
    eta_baseline=ig_result["eta_baseline"],
    feature_names=np.array(feature_cols),
)

logger.info("SHAP and IG attribution saved to %s", results_dir)
