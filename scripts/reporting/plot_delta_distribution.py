"""Figure: distribution of age gap Δ by age decile and sex.

Reads outputs/results/bioage_test.csv and saves a violin plot to
outputs/figures/delta_distribution.pdf.
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

# ---------------------------------------------------------------------------
# Snakemake / standalone dispatch
# ---------------------------------------------------------------------------

if "snakemake" in dir():
    _in_bioage = Path(snakemake.input[0])  # type: ignore[name-defined]
    _out_fig = Path(snakemake.output[0])  # type: ignore[name-defined]
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bioage", default="outputs/results/bioage_test.csv")
    parser.add_argument("--out", default="outputs/figures/delta_distribution.pdf")
    args = parser.parse_args()
    _in_bioage = Path(args.bioage)
    _out_fig = Path(args.out)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_out_fig.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(_in_bioage)
df["age_decile"] = pd.qcut(df["age_at_baseline"], 10, labels=False)

SEX_LABELS = {0: "Men", 1: "Women"}
COLORS = {0: "#2196F3", 1: "#E91E63"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
decile_centers = (
    df.groupby("age_decile")["age_at_baseline"].mean().values
)

for ax, sex_val in zip(axes, [0, 1]):
    sub = df[df["sex"] == sex_val]
    data_per_decile = [
        sub[sub["age_decile"] == d]["delta"].values
        for d in sorted(df["age_decile"].unique())
    ]
    parts = ax.violinplot(
        data_per_decile,
        positions=range(len(decile_centers)),
        showmedians=True,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS[sex_val])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(SEX_LABELS[sex_val], fontsize=12)
    ax.set_xlabel("Age decile (mean age, years)")
    ax.set_xticks(range(len(decile_centers)))
    ax.set_xticklabels([f"{c:.0f}" for c in decile_centers], rotation=45, ha="right")

axes[0].set_ylabel("Age gap Δ (years)")
fig.suptitle("Distribution of Biological Age Gap by Age Decile and Sex", fontsize=13)
fig.tight_layout()
fig.savefig(_out_fig, dpi=150)
plt.close(fig)
logger.info("Saved %s", _out_fig)
