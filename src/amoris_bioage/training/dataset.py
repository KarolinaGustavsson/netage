"""PyTorch Dataset for left-truncated survival data.

Each row represents one individual. The dataset returns tensors for the
network forward pass and for the Cox partial likelihood computation.
"""
from __future__ import annotations

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SurvivalDataset(Dataset):
    """Dataset for left-truncated survival data on the attained-age time scale.

    Individuals enter at ``age_at_baseline`` (left-truncation time) and exit
    at ``age_at_exit`` (event or censoring).  Both times are used as is — no
    conversion to follow-up duration.

    Args:
        df: Preprocessed DataFrame containing at minimum the columns
            ``age_at_baseline``, ``age_at_exit``, ``event``, ``sex``, and all
            columns listed in ``feature_cols``.
        feature_cols: Ordered list of column names to use as network inputs.
            Must match the ``n_features`` expected by the model.

    Each ``__getitem__`` call returns a dict with keys:

    * ``features``: ``(n_features,)`` float32 standardised biomarker values.
    * ``age``: scalar float32 age at baseline (= network age input and
      left-truncation time).
    * ``exit_time``: scalar float32 age at exit.
    * ``event``: scalar float32 binary event indicator (1 = death).
    * ``sex``: scalar int64 sex indicator (0 or 1).
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols not found in df: {missing}")

        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.age = torch.tensor(df["age_at_baseline"].values, dtype=torch.float32)
        self.exit_time = torch.tensor(df["age_at_exit"].values, dtype=torch.float32)
        self.event = torch.tensor(df["event"].values, dtype=torch.float32)
        self.sex = torch.tensor(df["sex"].values, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "features": self.features[idx],
            "age": self.age[idx],
            "exit_time": self.exit_time[idx],
            "event": self.event[idx],
            "sex": self.sex[idx],
        }
