"""Figure: Kaplan-Meier survival curves stratified by Δ decile.

Reads outputs/results/bioage_test.csv and saves KM curves to
outputs/figures/km_by_delta.pdf.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in dir():
    _in_bioage = Path(snakemake.input[0])  # type: ignore[name-defined]
    _out_fig = Path(snakemake.output[0])  # type: ignore[name-defined]
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bioage", default="outputs/results/bioage_test.csv")
    parser.add_argument("--out", default="outputs/figures/km_by_delta.pdf")
    args = parser.parse_args()
    _in_bioage = Path(args.bioage)
    _out_fig = Path(args.out)

_out_fig.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(_in_bioage)
# Follow-up from baseline.
df["follow_up"] = df["age_at_exit"] - df["age_at_baseline"]
df["delta_decile"] = pd.qcut(df["delta"], 10, labels=False, duplicates="drop")
n_deciles = df["delta_decile"].nunique()

cmap = plt.cm.RdYlBu_r  # low Δ = blue, high Δ = red

fig, ax = plt.subplots(figsize=(8, 5))

for d in sorted(df["delta_decile"].unique()):
    sub = df[df["delta_decile"] == d]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["follow_up"], sub["event"], label=f"Q{int(d) + 1}")
    color = cmap(d / max(n_deciles - 1, 1))
    kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=1.2)

ax.set_xlabel("Follow-up (years)")
ax.set_ylabel("Survival probability")
ax.set_title("Kaplan–Meier survival by age gap Δ decile\n(Q1 = lowest Δ, Q10 = highest Δ)")
ax.legend(loc="upper right", fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(_out_fig, dpi=150)
plt.close(fig)
logger.info("Saved %s", _out_fig)
