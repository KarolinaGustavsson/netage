"""Figure: group-level feature importance for g and Δ.

Groups features by biological domain (lipid, inflammation, renal, hepatic,
glycemic, hematologic) using configs/variable_groups.yaml and plots the
mean |SHAP| summed per group for both targets.

Reads shap_g.npz, shap_delta.npz, configs/variable_groups.yaml.
Writes outputs/figures/group_importance.pdf.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in dir():
    _in_g = Path(snakemake.input.shap_g)  # type: ignore[name-defined]
    _in_d = Path(snakemake.input.shap_delta)  # type: ignore[name-defined]
    _in_groups = Path(snakemake.input.groups)  # type: ignore[name-defined]
    _out_fig = Path(snakemake.output[0])  # type: ignore[name-defined]
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-g", default="outputs/results/shap_g.npz")
    parser.add_argument("--shap-delta", default="outputs/results/shap_delta.npz")
    parser.add_argument("--groups", default="configs/variable_groups.yaml")
    parser.add_argument("--out", default="outputs/figures/group_importance.pdf")
    args = parser.parse_args()
    _in_g = Path(args.shap_g)
    _in_d = Path(args.shap_delta)
    _in_groups = Path(args.groups)
    _out_fig = Path(args.out)

_out_fig.parent.mkdir(parents=True, exist_ok=True)

data_g = np.load(_in_g, allow_pickle=True)
data_d = np.load(_in_d, allow_pickle=True)
feat_names = data_g["feature_names"].tolist()
mean_abs_g = np.abs(data_g["values"]).mean(axis=0)
mean_abs_d = np.abs(data_d["values"]).mean(axis=0)

with open(_in_groups) as f:
    groups: dict = yaml.safe_load(f)

group_g: dict[str, float] = {}
group_d: dict[str, float] = {}
for group_name, members in groups.items():
    idxs = [feat_names.index(m) for m in members if m in feat_names]
    group_g[group_name] = float(mean_abs_g[idxs].sum()) if idxs else 0.0
    group_d[group_name] = float(mean_abs_d[idxs].sum()) if idxs else 0.0

group_names = list(group_g.keys())
x = np.arange(len(group_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_g = ax.bar(x - width / 2, [group_g[g] for g in group_names],
                width, label="Biological age g", color="#1565C0", alpha=0.85)
bars_d = ax.bar(x + width / 2, [group_d[g] for g in group_names],
                width, label="Age gap Δ", color="#C62828", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([g.capitalize() for g in group_names], rotation=30, ha="right")
ax.set_ylabel(r"Sum of mean |SHAP|")
ax.set_title("Group-level feature importance: g vs Δ", fontsize=12)
ax.legend()
fig.tight_layout()
fig.savefig(_out_fig, dpi=150)
plt.close(fig)
logger.info("Saved %s", _out_fig)
