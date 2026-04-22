"""Integrated gradients on the Cox network log-hazard η.

Captum's IntegratedGradients requires an end-to-end differentiable path.
Because g and Δ pass through the non-differentiable Breslow step-function
lookup, IG is applied to η(x, t) — the raw network output — rather than to
g or Δ directly.

Relationship to g/Δ attributions
----------------------------------
Since g is a monotone increasing function of η for fixed (t, sex) (higher
risk → older biological age), the sign of each feature's attribution on g is
identical to its attribution on η.  The magnitude is scaled by dg/dη, which
can be computed numerically and applied post-hoc if needed.

IG completeness (Axiomatic Attribution theorem):
    sum_j IG_j(x) = η(x) − η(baseline)

This is verified in ``tests/test_attribution.py``.

The IG baseline is the mean of the age-stratified background feature vectors,
matching the SHAP baseline convention so that both methods are comparable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from amoris_bioage.models.network import CoxMLP

logger = logging.getLogger(__name__)


class CoxIGExplainer:
    """Integrated gradients on the CoxMLP log-hazard output η.

    Args:
        model: Trained CoxMLP in eval mode.
        feature_cols: Ordered biomarker feature column names.
        background: Age-stratified background DataFrame produced by
            ``make_age_stratified_background``.  The per-column mean is used
            as the IG integration baseline.
        device: PyTorch device for forward and backward passes.
    """

    def __init__(
        self,
        model: CoxMLP,
        feature_cols: list[str],
        background: pd.DataFrame,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.eval()
        self.feature_cols = feature_cols
        self.device = torch.device(device)

        bg_feats = background[feature_cols].values.astype(np.float64)
        bg_ages = background["age_at_baseline"].values.astype(np.float64)

        self._baseline_features = torch.tensor(
            bg_feats.mean(axis=0, keepdims=True), dtype=torch.float32, device=self.device
        )
        self._baseline_age = torch.tensor(
            np.array([bg_ages.mean()]), dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        df: pd.DataFrame,
        n_steps: int = 50,
    ) -> dict[str, NDArray[np.float64]]:
        """Compute integrated gradients for each input with respect to η.

        Attributions for features and age are returned separately.

        IG completeness holds:
            feature_attributions[i].sum() + age_attribution[i]
            == η(x_i) − η(baseline)

        Args:
            df: Preprocessed DataFrame with ``feature_cols`` and
                ``age_at_baseline`` columns.
            n_steps: Number of Riemann sum steps for integration (default 50).
                Higher values give more accurate attributions.

        Returns:
            Dict with keys:

            * ``"feature_attributions"``: (N, n_features) IG values for
              biomarker features.
            * ``"age_attribution"``: (N,) IG value for chronological age.
            * ``"eta"``: (N,) network output η(x) for each individual.
            * ``"eta_baseline"``: scalar η(baseline).
        """
        from captum.attr import IntegratedGradients

        N = len(df)
        features = torch.tensor(
            df[self.feature_cols].values, dtype=torch.float32, device=self.device
        )
        age = torch.tensor(
            df["age_at_baseline"].values, dtype=torch.float32, device=self.device
        )

        baseline_feats = self._baseline_features.expand(N, -1)
        baseline_age = self._baseline_age.expand(N)

        ig = IntegratedGradients(self._forward)
        attr_feats, attr_age = ig.attribute(
            (features, age),
            baselines=(baseline_feats, baseline_age),
            n_steps=n_steps,
        )

        # Compute η values for completeness diagnostics.
        self.model.eval()
        with torch.no_grad():
            eta = self.model(features, age).cpu().numpy()
            eta_baseline = self.model(
                self._baseline_features, self._baseline_age[:1]
            ).item()

        return {
            "feature_attributions": attr_feats.cpu().detach().numpy(),
            "age_attribution": attr_age.cpu().detach().numpy(),
            "eta": eta,
            "eta_baseline": np.float64(eta_baseline),
        }

    def _forward(
        self, features: torch.Tensor, age: torch.Tensor
    ) -> torch.Tensor:
        return self.model(features, age)  # (N,)
