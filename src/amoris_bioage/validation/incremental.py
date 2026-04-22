"""Incremental C-index of the age gap Δ over chronological age.

Two Cox proportional hazards models are fitted on held-out data:

* **Null model**: Cox(age_at_baseline)
* **Full model**: Cox(age_at_baseline, Δ)

The likelihood ratio test (LRT) statistic is:

    Λ = 2 · (ℓ_full − ℓ_null)  ~  χ²(1)  under the null

A significant LRT (p < 0.05) means Δ captures mortality information beyond
what chronological age alone explains.

The incremental C-index Δ_C = C_full − C_null quantifies the discrimination
gained by including Δ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from numpy.typing import NDArray


def incremental_cindex_lrt(
    delta: NDArray[np.float64],
    age: NDArray[np.float64],
    event_times: NDArray[np.float64],
    events: NDArray[np.int64],
    entry_times: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Incremental C-index of Δ over chronological age and likelihood ratio test.

    Args:
        delta: (N,) age gap Δ(x, t) = g(x, t) − t in years.
        age: (N,) chronological age at baseline (age_at_baseline).
        event_times: (N,) age at exit on the attained-age scale (age_at_exit).
        events: (N,) binary event indicators (1 = death, 0 = censored).
        entry_times: (N,) left-truncation times.  When ``None``, left-
            truncation is ignored (equivalent to setting all entry times to 0).
            Pass ``age`` to use the standard attained-age left-truncation.

    Returns:
        Dict with keys:

        * ``c_null``: C-index of the age-only Cox model.
        * ``c_full``: C-index of the age + Δ Cox model.
        * ``delta_c``: Incremental C-index (c_full − c_null).
        * ``lrt_stat``: Likelihood ratio test statistic (chi-squared, 1 df).
        * ``p_value``: Two-tailed p-value from χ²(1) distribution.
        * ``ll_null``: Log partial likelihood of the null model.
        * ``ll_full``: Log partial likelihood of the full model.
    """
    delta = np.asarray(delta, dtype=np.float64)
    age = np.asarray(age, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)

    df = pd.DataFrame(
        {
            "T": event_times,
            "E": events,
            "age": age,
            "delta": delta,
        }
    )

    fit_kwargs: dict = dict(duration_col="T", event_col="E")
    if entry_times is not None:
        df["entry"] = np.asarray(entry_times, dtype=np.float64)
        fit_kwargs["entry_col"] = "entry"

    cph_null = CoxPHFitter()
    cph_null.fit(df[["T", "E", "age"] + (["entry"] if "entry" in df.columns else [])], **fit_kwargs)

    cph_full = CoxPHFitter()
    cph_full.fit(df, **fit_kwargs)

    ll_null = float(cph_null.log_likelihood_)
    ll_full = float(cph_full.log_likelihood_)
    lrt_stat = max(0.0, 2.0 * (ll_full - ll_null))
    p_value = float(scipy.stats.chi2.sf(lrt_stat, df=1))

    lh_null = cph_null.predict_log_partial_hazard(df[["T", "E", "age"] + (["entry"] if "entry" in df.columns else [])]).values
    lh_full = cph_full.predict_log_partial_hazard(df).values

    c_null = float(concordance_index(event_times, -lh_null, events))
    c_full = float(concordance_index(event_times, -lh_full, events))

    return {
        "c_null": c_null,
        "c_full": c_full,
        "delta_c": c_full - c_null,
        "lrt_stat": lrt_stat,
        "p_value": p_value,
        "ll_null": ll_null,
        "ll_full": ll_full,
    }
