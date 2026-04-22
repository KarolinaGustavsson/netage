"""Calibration of predicted 10-year mortality by decile of predicted risk.

Observed mortality within each bin is estimated using the Kaplan-Meier
product-limit estimator, which correctly handles individuals censored before
the prediction horizon.

Usage::

    df = calibration_by_decile(predicted_mortality, follow_up_times, events)
    # df has columns: bin, mean_predicted, observed_mortality, n, n_events
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from numpy.typing import NDArray


def calibration_by_decile(
    predicted_mortality: NDArray[np.float64],
    follow_up_times: NDArray[np.float64],
    events: NDArray[np.int64],
    horizon: float = 10.0,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calibration table: predicted vs observed horizon-year mortality by bin.

    Individuals are sorted by predicted mortality and divided into equal-
    frequency bins.  Within each bin, observed mortality at ``horizon`` years
    is estimated via the Kaplan-Meier estimator applied to follow-up time
    clipped at the horizon.

    Args:
        predicted_mortality: (N,) predicted probability of dying within
            ``horizon`` years, in [0, 1].
        follow_up_times: (N,) time from baseline to event or censoring
            (``age_at_exit − age_at_baseline``).
        events: (N,) binary event indicators (1 = death, 0 = censored).
        horizon: Prediction horizon in years (default 10).
        n_bins: Number of equal-frequency bins (default 10 = deciles).

    Returns:
        DataFrame with one row per bin and columns:

        * ``bin``: zero-based bin index.
        * ``mean_predicted``: mean predicted mortality within the bin.
        * ``observed_mortality``: KM estimate of mortality at ``horizon`` years.
        * ``n``: number of individuals in the bin.
        * ``n_events``: number of events within the horizon in the bin.
    """
    predicted_mortality = np.asarray(predicted_mortality, dtype=np.float64)
    follow_up_times = np.asarray(follow_up_times, dtype=np.float64)
    events = np.asarray(events)

    df = pd.DataFrame(
        {
            "pred": predicted_mortality,
            "ft": follow_up_times,
            "event": events.astype(int),
        }
    )
    df["bin"] = pd.qcut(df["pred"], q=n_bins, labels=False, duplicates="drop")

    rows = []
    for b in sorted(df["bin"].dropna().unique()):
        sub = df[df["bin"] == b].reset_index(drop=True)

        # Clip follow-up at the horizon; events after the horizon become censored.
        ft_clipped = np.minimum(sub["ft"].values, horizon)
        ev_clipped = (sub["event"].values == 1) & (sub["ft"].values <= horizon)

        kmf = KaplanMeierFitter()
        kmf.fit(ft_clipped, ev_clipped, label="km")
        # S(horizon) is the estimated probability of surviving beyond the horizon.
        # KaplanMeierFitter.predict() returns the survival probability at the
        # given time(s); we subtract from 1 to get mortality.
        survival_at_horizon = float(kmf.predict(horizon))
        observed_mortality = 1.0 - survival_at_horizon

        rows.append(
            {
                "bin": int(b),
                "mean_predicted": float(sub["pred"].mean()),
                "observed_mortality": observed_mortality,
                "n": len(sub),
                "n_events": int(ev_clipped.sum()),
            }
        )

    return pd.DataFrame(rows)
