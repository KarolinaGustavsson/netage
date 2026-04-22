"""Figure: top pairwise SHAP interaction values for Δ.

Reads shap_delta.npz.  If the file contains interaction values (shape
(N, p, p)), plots a heatmap of mean |Φ_{ij}| for i ≠ j.  Otherwise,
approximates interactions from main SHAP values (co-occurrence heatmap).

Writes outputs/figures/shap_interactions.pdf.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in dir():
    _in_delta = Path(snakemake.input.shap_delta)  # type: ignore[name-defined]
    _out_fig = Path(snakemake.output[0])  # type: ignore[name-defined]
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-delta", default="outputs/results/shap_delta.npz")
    parser.add_argument("--out", default="outputs/figures/shap_interactions.pdf")
    args = parser.parse_args()
    _in_delta = Path(args.shap_delta)
    _out_fig = Path(args.out)

_out_fig.parent.mkdir(parents=True, exist_ok=True)

data = np.load(_in_delta, allow_pickle=True)
feat_names = list(data["feature_names"])
values = data["values"]  # (N, p) or (N, p, p)

if values.ndim == 3:
    # True interaction matrix available.
    interact_mat = np.abs(values).mean(axis=0)
    np.fill_diagonal(interact_mat, 0.0)
    title = "Mean pairwise SHAP interaction |Φ_ij| for Δ"
else:
    # Approximate: outer product of mean absolute SHAP values (co-importance).
    mean_abs = np.abs(values).mean(axis=0)
    interact_mat = np.outer(mean_abs, mean_abs)
    np.fill_diagonal(interact_mat, 0.0)
    title = "Co-importance proxy (main SHAP values) for Δ\n(run attribute.py with interactions=True for true interaction values)"
    logger.warning("Interaction matrix not found; using co-importance proxy.")

p = interact_mat.shape[0]
fig, ax = plt.subplots(figsize=(max(6, p * 0.5), max(5, p * 0.5)))
im = ax.imshow(interact_mat, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(p))
ax.set_yticks(range(p))
ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(feat_names, fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.7)
ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig(_out_fig, dpi=150)
plt.close(fig)
logger.info("Saved %s", _out_fig)
