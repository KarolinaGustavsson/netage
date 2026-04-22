"""Imputation and standardisation for the AMORIS feature matrix.

The Preprocessor is fit on the training set only; the same statistics are
applied unchanged to validation and test sets to prevent data leakage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from amoris_bioage.data.schema import FEATURE_COLS

logger = logging.getLogger(__name__)

# Columns with missingness above this fraction receive an additional binary
# indicator column so the model can distinguish imputed from observed values.
MISSINGNESS_INDICATOR_THRESHOLD: float = 0.05


@dataclass
class Preprocessor:
    """Median imputer and standard scaler for AMORIS biomarker features.

    Fit on training data; apply the same transforms to validation/test splits.

    Attributes:
        feature_cols: Ordered list of feature column names to process.
    """

    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))

    _medians: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _means: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _stds: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _high_missingness_cols: list[str] = field(default_factory=list, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Compute imputation and scaling statistics from training data.

        Args:
            df: Training set DataFrame; must contain all columns in
                ``self.feature_cols``.

        Returns:
            self, to allow chaining with ``transform``.
        """
        self._high_missingness_cols = []
        for col in self.feature_cols:
            observed = df[col].dropna()
            self._medians[col] = float(observed.median())
            self._means[col] = float(observed.mean())
            self._stds[col] = float(observed.std(ddof=1))

            miss_rate = df[col].isna().mean()
            if miss_rate > MISSINGNESS_INDICATOR_THRESHOLD:
                self._high_missingness_cols.append(col)
                logger.debug(
                    "Column %s: %.1f%% missing — adding indicator column",
                    col,
                    miss_rate * 100,
                )

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values and standardise features.

        Args:
            df: DataFrame to transform. May be the training, validation, or
                test set; scaling statistics always come from the fit call.

        Returns:
            New DataFrame with imputed and z-scored features. Missing-value
            indicator columns are appended for high-missingness features.
            The input DataFrame is not modified.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor.fit() must be called before transform().")

        out = df.copy()

        for col in self.feature_cols:
            if col in self._high_missingness_cols:
                out[f"{col}_missing"] = out[col].isna().astype(float)

            out[col] = out[col].fillna(self._medians[col])

            std = self._stds[col]
            if std > 0:
                out[col] = (out[col] - self._means[col]) / std
            else:
                out[col] = 0.0

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` then transform it.

        Args:
            df: Training set DataFrame.

        Returns:
            Transformed copy of ``df``.
        """
        return self.fit(df).transform(df)
