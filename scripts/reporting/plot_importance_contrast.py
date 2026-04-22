"""Figure: importance contrast — mean |φ^{BA}| vs mean |φ^{Δ}| per feature.

Reads shap_g.npz and shap_delta.npz and saves a scatter/dot-plot to
outputs/figures/importance_contrast.pdf.

Each point is one feature.  Features above the diagonal contribute more to
biological age g than to the age gap Δ; features below contribute more to Δ.
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
    _in_g = Path(snakemake.input.shap_g)  # type: ignore[name-defined]
    _in_d = Path(snakemake.input.shap_delta)  # type: ignore[name-defined]
    _out_fig = Path(snakemake.output[0])  # type: ignore[name-defined]
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-g", default="outputs/results/shap_g.npz")
    parser.add_argument("--shap-delta", default="outputs/results/shap_delta.npz")
    parser.add_argument("--out", default="outputs/figures/importance_contrast.pdf")
    args = parser.parse_args()
    _in_g = Path(args.shap_g)
    _in_d = Path(args.shap_delta)
    _out_fig = Path(args.out)

_out_fig.parent.mkdir(parents=True, exist_ok=True)

data_g = np.load(_in_g, allow_pickle=True)
data_d = np.load(_in_d, allow_pickle=True)

feat_names = data_g["feature_names"].tolist()
mean_abs_g = np.abs(data_g["values"]).mean(axis=0)
mean_abs_d = np.abs(data_d["values"]).mean(axis=0)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(mean_abs_g, mean_abs_d, color="#1565C0", edgecolors="white", s=60, zorder=3)

# Diagonal reference line (equal importance for g and Δ).
lim = max(mean_abs_g.max(), mean_abs_d.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="Equal importance")

# Label each feature.
for name, x, y in zip(feat_names, mean_abs_g, mean_abs_d):
    ax.annotate(name, (x, y), fontsize=7, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel(r"Mean |$\phi^{BA}$| (biological age)", fontsize=11)
ax.set_ylabel(r"Mean |$\phi^{\Delta}$| (age gap)", fontsize=11)
ax.set_title("Feature importance contrast: g vs Δ", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(_out_fig, dpi=150)
plt.close(fig)
logger.info("Saved %s", _out_fig)
