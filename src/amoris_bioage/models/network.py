"""Feedforward Cox network η_θ(x, t) that outputs a log-hazard ratio.

Input:  15 standardised biomarker features + chronological age at baseline.
Output: scalar η(x, t) — the individual log-hazard ratio relative to H_0(t).

The Cox hazard for individual i at time t is:
    h(t | x_i) = h_0(t) · exp(η_θ(x_i, t_i))

where h_0(t) is the Breslow baseline hazard estimated on the training set.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
}


class CoxMLP(nn.Module):
    """Two-to-three hidden layer MLP for the Cox log-hazard ratio.

    Args:
        n_features: Number of biomarker covariates, not counting age.
            Default 15 matches the AMORIS feature set.
        hidden_sizes: Width of each hidden layer; 2–3 entries per CLAUDE.md.
        dropout: Dropout probability applied after each activation layer.
        activation: Non-linearity: "relu" (default), "elu", or "selu".
    """

    def __init__(
        self,
        n_features: int = 15,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
        activation: Literal["relu", "elu", "selu"] = "relu",
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(_ACTIVATIONS)}, got {activation!r}"
            )

        act_cls = _ACTIVATIONS[activation]
        in_size = n_features + 1  # features + age

        layers: list[nn.Module] = []
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), act_cls(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.net = nn.Sequential(*layers)
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, features: Tensor, age: Tensor) -> Tensor:
        """Compute per-individual log-hazard ratios.

        Args:
            features: (N, n_features) standardised biomarker matrix.
            age: (N,) chronological age at baseline, in years.

        Returns:
            (N,) log-hazard ratio η(x, t).
        """
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {features.shape[-1]}"
            )
        x = torch.cat([features, age.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)
