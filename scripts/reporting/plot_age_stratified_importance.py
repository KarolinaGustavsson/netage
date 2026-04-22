"""Figure: age-stratified importance for the top features (by mean |φ^Δ|).

Shows how each top feature's mean |SHAP| on Δ changes across age deciles.
This reveals whether certain features drive biological ageing more strongly
at specific life stages.

Reads shap_delta.npz and bioage_test.csv (for age bins).
Writes outputs/figures/age_stratified_importance.pdf.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TOP_N = 6  # number of top features to plot

if "snakemake" in dir():
    _in_delta = Path(snakemake.input.shap_delta)  # type: ignore[name-defined]
    _in_bioage = Path(snakemake.input.bioage)  # type: ignore[name-defined]
    _out_fig = Path(snakemake.output[0])  # type: ignore[name-defined]
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-delta", default="outputs/results/shap_delta.npz")
    parser.add_argument("--bioage", default="outputs/results/bioage_test.csv")
    parser.add_argument("--out", default="outputs/figures/age_stratified_importance.pdf")
    args = parser.parse_args()
    _in_delta = Path(args.shap_delta)
    _in_bioage = Path(args.bioage)
    _out_fig = Path(args.out)

_out_fig.parent.mkdir(parents=True, exist_ok=True)

data = np.load(_in_delta, allow_pickle=True)
feat_names = data["feature_names"].tolist()
shap_vals = data["values"]  # (N, n_features)

bioage_df = pd.read_csv(_in_bioage).reset_index(drop=True)
# Align by position (shap subset may be sampled; use index order).
n = min(len(shap_vals), len(bioage_df))
ages = bioage_df["age_at_baseline"].values[:n]
shap_vals = shap_vals[:n]

# Select top features by global mean |SHAP|.
mean_abs = np.abs(shap_vals).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:TOP_N]
top_names = [feat_names[i] for i in top_idx]

# Age decile bins.
age_bins = pd.qcut(ages, 10, labels=False, duplicates="drop")
decile_centers = pd.Series(ages).groupby(age_bins).mean().values

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
axes = axes.flatten()

for ax, feat_i, name in zip(axes, top_idx, top_names):
    vals_per_decile = [
        np.abs(shap_vals[age_bins == d, feat_i])
        for d in sorted(np.unique(age_bins[~np.isnan(age_bins)]).astype(int))
    ]
    centers = [decile_centers[d] for d in range(len(vals_per_decile))]
    means = [v.mean() if len(v) > 0 else 0.0 for v in vals_per_decile]
    stds = [v.std() if len(v) > 1 else 0.0 for v in vals_per_decile]

    ax.plot(centers, means, marker="o", linewidth=1.5, color="#1565C0")
    ax.fill_between(centers,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color="#1565C0")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Age (years)", fontsize=8)
    ax.set_ylabel(r"Mean |$\phi^\Delta$|", fontsize=8)

fig.suptitle(f"Age-stratified importance for top {TOP_N} features (Δ)", fontsize=12)
fig.tight_layout()
fig.savefig(_out_fig, dpi=150)
plt.close(fig)
logger.info("Saved %s", _out_fig)
