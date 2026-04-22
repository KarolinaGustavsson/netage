"""Figure: comparator benchmark — incremental C-index of Δ over age.

Reads outputs/results/metrics.json (and optionally comparator metric files)
and produces a grouped bar chart comparing:

  - Neural Cox (this model)
  - Linear Cox PhenoAge (if comparator metrics exist)
  - Klemera–Doubal (if comparator metrics exist)

Writes outputs/figures/comparator_benchmark.pdf.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in dir():
    _in_metrics = Path(snakemake.input.metrics)  # type: ignore[name-defined]
    _out_fig = Path(snakemake.output[0])  # type: ignore[name-defined]
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/results/metrics.json")
    parser.add_argument("--out", default="outputs/figures/comparator_benchmark.pdf")
    args = parser.parse_args()
    _in_metrics = Path(args.metrics)
    _out_fig = Path(args.out)

_out_fig.parent.mkdir(parents=True, exist_ok=True)

with open(_in_metrics) as f:
    metrics = json.load(f)

lrt = metrics["incremental_lrt"]
models = ["Neural Cox"]
c_null = [lrt["c_null"]]
c_full = [lrt["c_full"]]
delta_c = [lrt["delta_c"]]
p_values = [lrt["p_value"]]

# Load comparator metrics if they exist.
for name, path in [
    ("Linear Cox PhenoAge", Path("outputs/results/metrics_linear_cox.json")),
    ("Klemera–Doubal", Path("outputs/results/metrics_kd.json")),
]:
    if path.exists():
        with open(path) as f:
            comp = json.load(f)
        models.append(name)
        c_null.append(comp["incremental_lrt"]["c_null"])
        c_full.append(comp["incremental_lrt"]["c_full"])
        delta_c.append(comp["incremental_lrt"]["delta_c"])
        p_values.append(comp["incremental_lrt"]["p_value"])

x = np.arange(len(models))
width = 0.35
colors_null = ["#90CAF9"] * len(models)
colors_full = ["#1565C0", "#C62828", "#2E7D32"][: len(models)]

fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.5), 5))
ax.bar(x - width / 2, c_null, width, label="Cox(age)", color="#90CAF9")
ax.bar(x + width / 2, c_full, width, label="Cox(age + Δ)",
       color=["#1565C0", "#C62828", "#2E7D32"][: len(models)])

# Annotate with Δ_C and significance.
for i, (dc, pv) in enumerate(zip(delta_c, p_values)):
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
    ax.text(x[i] + width / 2, c_full[i] + 0.003,
            f"Δ={dc:+.3f}\n{sig}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylabel("C-index")
ax.set_ylim(max(0.4, min(c_null) - 0.05), min(1.0, max(c_full) + 0.08))
ax.set_title("Incremental C-index of Δ over chronological age", fontsize=12)
ax.legend()
fig.tight_layout()
fig.savefig(_out_fig, dpi=150)
plt.close(fig)
logger.info("Saved %s", _out_fig)
