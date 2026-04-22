"""Cox partial likelihood with Efron tie correction and left-truncation.

Left-truncation (attained-age time scale)
-----------------------------------------
Individual i enters the risk set at age_at_baseline_i and exits at age_at_exit_i.
At any attained age t, the risk set is:
    R(t) = { i : entry_i < t  AND  exit_i >= t }

This is implemented exactly — there is no approximation for the risk set.

Efron tie correction
--------------------
When d individuals die at the same attained age t, the partial likelihood
contribution is:

    log L(t) = Σ_{i∈D(t)} η_i
               - Σ_{l=0}^{d-1} log( S_R(t) − (l/d) · S_D(t) )

where S_R(t) = Σ_{i∈R(t)} exp(η_i)  and  S_D(t) = Σ_{i∈D(t)} exp(η_i).

For d = 1 (no ties) this reduces to the standard partial likelihood.

References
----------
Efron, B. (1977). The efficiency of Cox's likelihood function for censored
data. JASA, 72(359), 557-565.
"""
from __future__ import annotations

import torch
from torch import Tensor


def cox_partial_likelihood_efron(
    log_hazard: Tensor,
    event_times: Tensor,
    events: Tensor,
    entry_times: Tensor,
) -> Tensor:
    """Negative mean log partial likelihood with Efron tie correction.

    All input tensors must reside on the same device and share a floating-
    point dtype. Gradients flow through ``log_hazard``.

    Args:
        log_hazard: (N,) network output η(x, t).
        event_times: (N,) age_at_exit on the attained-age time scale.
        events: (N,) binary event indicators (1 = death, 0 = censored).
            Must be castable to bool.
        entry_times: (N,) age_at_baseline, i.e., the left-truncation times.

    Returns:
        Scalar: negative log partial likelihood divided by the number of
        events (for consistent gradient magnitude across batches of different
        sizes).

    Raises:
        ValueError: If the batch contains no events.
    """
    event_mask = events.bool()
    n_events = event_mask.sum()
    if n_events == 0:
        raise ValueError("Batch contains no events; cannot compute Cox partial likelihood.")

    # Unique event times in ascending order.
    unique_times = event_times[event_mask].unique()

    log_lik: Tensor = log_hazard.new_zeros(())

    for t in unique_times:
        # Left-truncated risk set: entry < t ≤ exit.
        in_risk = (entry_times < t) & (event_times >= t)
        is_death = event_mask & (event_times == t)

        d = int(is_death.sum().item())
        if d == 0 or not in_risk.any():
            continue

        log_hz_risk = log_hazard[in_risk]
        log_hz_death = log_hazard[is_death]

        # Numerator: sum of log-hazards for events at t.
        numerator = log_hz_death.sum()

        # Efron denominator using logsumexp for numerical stability.
        S_R = torch.exp(torch.logsumexp(log_hz_risk, dim=0))
        S_D = torch.exp(torch.logsumexp(log_hz_death, dim=0))

        efron_log_denom: Tensor = log_hazard.new_zeros(())
        for l in range(d):
            alpha = l / d
            # S_R - alpha*S_D >= 0 because alpha < 1 and all exp values are positive.
            denom = S_R - alpha * S_D
            efron_log_denom = efron_log_denom + torch.log(denom.clamp(min=1e-30))

        log_lik = log_lik + numerator - efron_log_denom

    return -log_lik / n_events
