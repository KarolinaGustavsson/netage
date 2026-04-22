"""Nonparametric Breslow cumulative baseline hazard, sex-stratified.

The estimator respects the left-truncated risk set on the attained-age time
scale: individual i is at risk at time t iff entry_i < t ≤ exit_i.

The increment at each unique event time t_j for stratum s is:

    ΔĤ_0^s(t_j) = d_j^s / Σ_{i ∈ R_j^s} exp(η_i)

where d_j^s is the number of events at t_j in stratum s and R_j^s is the
left-truncated risk set restricted to stratum s.

Cumulative hazard is evaluated by step-function (right-continuous) lookup.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class BreslowEstimator:
    """Sex-stratified Breslow cumulative baseline hazard estimator.

    Fit on the training set after the network has been trained. The same
    network log-hazard values used for training should be passed to ``fit``.

    After fitting, ``predict_cumhaz`` evaluates the cumulative baseline
    hazard Ĥ_0(t) via right-continuous step-function interpolation.
    """

    def __init__(self) -> None:
        # Per-stratum sorted event times and cumulative baseline hazard.
        self._event_times: dict[int, NDArray[np.float64]] = {}
        self._cumhaz: dict[int, NDArray[np.float64]] = {}
        self._is_fitted: bool = False

    def fit(
        self,
        log_hazard: NDArray[np.float64],
        event_times: NDArray[np.float64],
        events: NDArray[np.int64],
        entry_times: NDArray[np.float64],
        sex: NDArray[np.int64],
    ) -> "BreslowEstimator":
        """Fit sex-stratified Breslow estimator.

        Args:
            log_hazard: (N,) η(x, t) from the trained network.
            event_times: (N,) age_at_exit (attained-age time scale).
            events: (N,) binary event indicator (1 = death, 0 = censored).
            entry_times: (N,) age_at_baseline (left-truncation times).
            sex: (N,) sex indicator (0 or 1).

        Returns:
            self
        """
        log_hazard = np.asarray(log_hazard, dtype=np.float64)
        event_times = np.asarray(event_times, dtype=np.float64)
        events = np.asarray(events, dtype=np.int64)
        entry_times = np.asarray(entry_times, dtype=np.float64)
        sex = np.asarray(sex, dtype=np.int64)

        for s in [0, 1]:
            mask = sex == s
            if not mask.any():
                raise ValueError(f"No individuals with sex={s} in training data")
            n_s = mask.sum()
            n_events_s = events[mask].sum()
            logger.debug("Fitting Breslow stratum sex=%d: n=%d events=%d", s, n_s, n_events_s)
            self._fit_stratum(
                log_hazard[mask],
                event_times[mask],
                events[mask],
                entry_times[mask],
                stratum=s,
            )

        self._is_fitted = True
        return self

    def _fit_stratum(
        self,
        log_hazard: NDArray[np.float64],
        event_times: NDArray[np.float64],
        events: NDArray[np.int64],
        entry_times: NDArray[np.float64],
        stratum: int,
    ) -> None:
        unique_times = np.unique(event_times[events == 1])
        exp_eta = np.exp(log_hazard)

        increments = np.zeros(len(unique_times), dtype=np.float64)
        for j, t in enumerate(unique_times):
            # Left-truncated risk set for this stratum.
            in_risk = (entry_times < t) & (event_times >= t)
            d_j = int(((event_times == t) & (events == 1)).sum())
            denom = exp_eta[in_risk].sum()
            if denom > 0 and d_j > 0:
                increments[j] = d_j / denom

        self._event_times[stratum] = unique_times
        self._cumhaz[stratum] = np.cumsum(increments)

    def predict_cumhaz(
        self,
        times: NDArray[np.float64],
        sex: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Evaluate cumulative baseline hazard at arbitrary attained ages.

        Uses right-continuous step-function interpolation: returns the
        cumulative hazard at the last observed event time ≤ t. Returns 0
        for times before the first event, and the maximum cumulative hazard
        for times beyond the last event.

        Args:
            times: (N,) ages at which to evaluate Ĥ_0.
            sex: (N,) sex indicator (0 or 1).

        Returns:
            (N,) cumulative baseline hazard values.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError("BreslowEstimator must be fitted before predict_cumhaz.")

        times = np.asarray(times, dtype=np.float64)
        sex = np.asarray(sex, dtype=np.int64)
        result = np.zeros(len(times), dtype=np.float64)

        for s in [0, 1]:
            mask = sex == s
            if not mask.any():
                continue
            result[mask] = _step_interp(
                times[mask], self._event_times[s], self._cumhaz[s]
            )
        return result


def _step_interp(
    query: NDArray[np.float64],
    knots: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Right-continuous step function: value at the largest knot ≤ query.

    Returns 0 for query < knots[0] and values[-1] for query > knots[-1].
    """
    idx = np.searchsorted(knots, query, side="right") - 1
    # idx == -1 when query < knots[0]: clamp to 0 for lookup, then zero out.
    before_first = idx < 0
    idx_safe = np.maximum(idx, 0)
    out = values[idx_safe]
    out[before_first] = 0.0
    return out
