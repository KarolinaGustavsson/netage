"""SHAP attribution for biological age g and age gap Δ.

Computes Shapley values for the full pipeline (network → Breslow → inversion),
treating g(x, t) and Δ(x, t) = g(x, t) − t as black-box scalar-valued
functions of the input (features, age, sex).

The background distribution is an age-stratified cohort-mean set from
``make_age_stratified_background``.  Attributions are therefore relative to
the age-matched average individual, which matches the biological age
definition's reference population.

Sex is included in the SHAP input array as a binary covariate.  It is snapped
to {0, 1} inside the prediction wrapper so that SHAP can treat it as any other
numeric input while the Breslow estimator always receives an integer value.

Notes
-----
SHAP interaction values for general black-box models are available via
``shap.Explainer(..., interactions=True)`` but scale quadratically in the
number of features.  Use a subset of the test set for interaction computation.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import shap
import torch
from numpy.typing import NDArray

from amoris_bioage.bioage.breslow import BreslowEstimator
from amoris_bioage.bioage.inversion import BiologicalAgeEstimator
from amoris_bioage.models.network import CoxMLP

logger = logging.getLogger(__name__)

_EXTRA_COLS = ["age_at_baseline", "sex"]


class BioageShapExplainer:
    """SHAP explainer for g(x, t) and Δ(x, t).

    Wraps the full Cox pipeline as a model-agnostic prediction function and
    applies ``shap.Explainer`` with an ``Independent`` masker over the
    age-stratified background.

    Args:
        model: Trained CoxMLP in eval mode.
        breslow: Fitted BreslowEstimator on the training set.
        bioage_estimator: Fitted BiologicalAgeEstimator.
        feature_cols: Ordered biomarker feature column names.  Must match the
            columns the model was trained on (excluding age).
        background: Age-stratified background DataFrame from
            ``make_age_stratified_background``.  Must contain
            ``feature_cols + ["age_at_baseline", "sex"]``.
        device: PyTorch device for the network forward pass.
    """

    def __init__(
        self,
        model: CoxMLP,
        breslow: BreslowEstimator,
        bioage_estimator: BiologicalAgeEstimator,
        feature_cols: list[str],
        background: pd.DataFrame,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.eval()
        self.breslow = breslow
        self.bioage_estimator = bioage_estimator
        self.feature_cols = feature_cols
        self.input_cols = feature_cols + _EXTRA_COLS
        self.device = torch.device(device)

        missing = [c for c in self.input_cols if c not in background.columns]
        if missing:
            raise ValueError(f"Background is missing columns: {missing}")
        self._bg_array = background[self.input_cols].values.astype(np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_delta(
        self,
        df: pd.DataFrame,
        max_evals: int | str = "auto",
    ) -> shap.Explanation:
        """Compute SHAP values for Δ(x, t).

        Args:
            df: DataFrame with columns ``self.input_cols``.
            max_evals: Maximum number of model evaluations per sample.  Higher
                values give more accurate SHAP estimates.  Pass ``"auto"`` to
                let shap decide.

        Returns:
            ``shap.Explanation`` with ``.values`` of shape (N, n_input_cols)
            and ``.base_values`` of shape (N,).  The columns correspond to
            ``self.input_cols`` (features + age_at_baseline + sex).
            Shapley completeness: ``values[i].sum() + base_values[i] == Δ[i]``.
        """
        return self._explain(df, target="delta", max_evals=max_evals)

    def explain_g(
        self,
        df: pd.DataFrame,
        max_evals: int | str = "auto",
    ) -> shap.Explanation:
        """Compute SHAP values for g(x, t) (biological age).

        Args:
            df: DataFrame with columns ``self.input_cols``.
            max_evals: Maximum number of model evaluations per sample.

        Returns:
            ``shap.Explanation`` with ``.values`` of shape (N, n_input_cols)
            and ``.base_values`` of shape (N,).
            Shapley completeness: ``values[i].sum() + base_values[i] == g[i]``.
        """
        return self._explain(df, target="g", max_evals=max_evals)

    def explain_interactions(
        self,
        df: pd.DataFrame,
        max_evals: int | str = "auto",
    ) -> shap.Explanation:
        """Compute SHAP interaction values for Δ(x, t).

        Interaction values Φ_{ij} decompose the SHAP value of feature j into
        the portion attributable to j alone (Φ_{jj}) and the portion shared
        with each other feature i (Φ_{ij} = Φ_{ji}).

        Warning: scales quadratically with n_features.  Use a small subset
        of the test set (≤200 individuals).

        Args:
            df: DataFrame with columns ``self.input_cols``.
            max_evals: Maximum number of model evaluations per sample.

        Returns:
            ``shap.Explanation`` with ``.values`` of shape
            (N, n_input_cols, n_input_cols).
        """
        missing = [c for c in self.input_cols if c not in df.columns]
        if missing:
            raise ValueError(f"df missing columns: {missing}")

        X = df[self.input_cols].values.astype(np.float64)
        masker = shap.maskers.Independent(self._bg_array)
        explainer = shap.Explainer(
            self._predict_delta, masker, output_names=self.input_cols
        )
        kwargs = {} if max_evals == "auto" else {"max_evals": max_evals}
        return explainer(X, interactions=True, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _explain(
        self,
        df: pd.DataFrame,
        target: Literal["delta", "g"],
        max_evals: int | str = "auto",
    ) -> shap.Explanation:
        missing = [c for c in self.input_cols if c not in df.columns]
        if missing:
            raise ValueError(f"df missing columns: {missing}")

        X = df[self.input_cols].values.astype(np.float64)
        predict_fn = self._predict_delta if target == "delta" else self._predict_g

        masker = shap.maskers.Independent(self._bg_array)
        explainer = shap.Explainer(
            predict_fn, masker, output_names=self.input_cols
        )
        kwargs = {} if max_evals == "auto" else {"max_evals": max_evals}
        return explainer(X, **kwargs)

    def _predict_delta(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        _, delta = self._pipeline(X)
        return delta

    def _predict_g(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        g, _ = self._pipeline(X)
        return g

    def _pipeline(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Full pipeline: [features | age | sex] → (g, Δ)."""
        n_feat = len(self.feature_cols)
        features = torch.tensor(
            X[:, :n_feat], dtype=torch.float32, device=self.device
        )
        age = torch.tensor(X[:, n_feat], dtype=torch.float32, device=self.device)
        # Snap sex to {0, 1}; SHAP may pass non-integer values when masking.
        sex = np.clip(np.round(X[:, n_feat + 1]).astype(np.int64), 0, 1)

        with torch.no_grad():
            log_hz = self.model(features, age).cpu().numpy()

        g, delta = self.bioage_estimator.transform(
            log_hz, age.cpu().numpy(), sex, self.breslow
        )
        return g, delta
