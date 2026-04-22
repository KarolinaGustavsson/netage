"""Harrell's concordance index (C-index) for survival models.

The C-index measures the probability that, for a randomly chosen pair of
individuals where one died before the other, the model assigned the higher
predicted risk to the individual who died first.

Convention used throughout: ``log_hazard`` is the network output η(x, t)
where higher values mean higher risk (shorter expected survival).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_cindex(
    log_hazard: NDArray[np.float64],
    event_times: NDArray[np.float64],
    events: NDArray[np.int64],
) -> float:
    """Harrell's C-index for a Cox model.

    Args:
        log_hazard: (N,) predicted log-hazard ratios η(x, t).  Higher values
            imply higher risk.
        event_times: (N,) observed survival times (age at exit on the
            attained-age scale).
        events: (N,) binary event indicators (1 = death, 0 = censored).

    Returns:
        C-index in [0, 1].  0.5 corresponds to chance; 1.0 to perfect
        concordance (every death ranked as higher risk than every censoring
        at a later time).
    """
    from lifelines.utils import concordance_index

    log_hazard = np.asarray(log_hazard, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    events = np.asarray(events)

    # lifelines convention: higher predicted_score → longer expected survival.
    # We negate log_hazard so that lower log_hazard (lower risk) maps to a
    # higher predicted survival time.
    return float(concordance_index(event_times, -log_hazard, events))
