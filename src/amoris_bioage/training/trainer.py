"""Training loop for CoxMLP with AdamW optimiser and cosine LR schedule.

Mini-batch Cox partial likelihood is an approximation of the full-cohort
partial likelihood; it is the standard approach for large survival datasets
(Kvamme et al. 2019).  The approximation becomes exact when the batch covers
the entire training set.

Early stopping monitors validation C-index (Harrell's concordance statistic)
and restores the best checkpoint when patience is exhausted.

Time scale: attained age.  Entry time = age_at_baseline (left-truncation),
exit time = age_at_exit.  The Cox loss receives both so that the risk set
at each event time excludes individuals who have not yet entered the study.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from amoris_bioage.config import TrainingConfig
from amoris_bioage.models.cox_loss import cox_partial_likelihood_efron
from amoris_bioage.models.network import CoxMLP
from amoris_bioage.training.dataset import SurvivalDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Output of a completed training run.

    Attributes:
        model: CoxMLP with best-checkpoint weights loaded, in eval mode.
        history: Per-epoch records with keys ``epoch``, ``train_loss``,
            ``val_cindex``.
        best_epoch: Zero-based epoch index of the best validation C-index.
        best_val_cindex: Best validation C-index achieved during training.
    """

    model: CoxMLP
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_cindex: float = 0.0


class Trainer:
    """Training loop for CoxMLP.

    Args:
        model: Initialised CoxMLP.  Modified in-place during training.
        config: TrainingConfig specifying optimiser and schedule parameters.
        device: PyTorch device (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
        checkpoint_path: If provided, the best model state dict is saved here
            as a ``.pt`` file after each improvement.
    """

    def __init__(
        self,
        model: CoxMLP,
        config: TrainingConfig,
        device: str | torch.device = "cpu",
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        torch.manual_seed(config.seed)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> TrainResult:
        """Train the model and return the best checkpoint.

        Args:
            train_df: Preprocessed training DataFrame.  Must contain all
                columns in ``feature_cols`` plus ``age_at_baseline``,
                ``age_at_exit``, ``event``, and ``sex``.
            val_df: Preprocessed validation DataFrame with the same schema.
            feature_cols: Ordered list of feature column names passed to the
                network.  Typically ``FEATURE_COLS`` plus any missing-indicator
                columns added by the Preprocessor.

        Returns:
            TrainResult with the best model, per-epoch history, and metrics.
        """
        cfg = self.config
        model = self.model.to(self.device)

        train_ds = SurvivalDataset(train_df, feature_cols)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(cfg.seed),
            drop_last=False,
        )

        # Pre-load the full validation set as tensors for fast C-index scoring.
        val_ds = SurvivalDataset(val_df, feature_cols)
        val_features = val_ds.features.to(self.device)
        val_age = val_ds.age.to(self.device)
        val_exit = val_ds.exit_time.numpy()
        val_events = val_ds.event.numpy()

        optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=0.0)

        best_val_cindex: float = -1.0
        best_state: dict = copy.deepcopy(model.state_dict())
        best_epoch: int = 0
        patience_counter: int = 0
        history: list[dict] = []

        for epoch in range(cfg.max_epochs):
            train_loss = self._train_epoch(model, train_loader, optimizer)
            scheduler.step()
            val_cindex = self._val_cindex(
                model, val_features, val_age, val_exit, val_events
            )

            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_cindex": val_cindex}
            )
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_cindex=%.4f",
                epoch,
                train_loss,
                val_cindex,
            )

            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                if self.checkpoint_path is not None:
                    torch.save(best_state, self.checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d "
                        "(best epoch %d, val_cindex=%.4f)",
                        epoch,
                        best_epoch,
                        best_val_cindex,
                    )
                    break

        model.load_state_dict(best_state)
        model.eval()

        return TrainResult(
            model=model,
            history=history,
            best_epoch=best_epoch,
            best_val_cindex=best_val_cindex,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: CoxMLP,
        loader: DataLoader,
        optimizer: AdamW,
    ) -> float:
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            features = batch["features"].to(self.device)
            age = batch["age"].to(self.device)
            exit_time = batch["exit_time"].to(self.device)
            event = batch["event"].to(self.device)

            # Skip batches that contain no events (loss is undefined).
            if event.sum() == 0:
                continue

            optimizer.zero_grad()
            log_hz = model(features, age)
            # entry_times = age_at_baseline (left-truncation on attained-age scale).
            loss = cox_partial_likelihood_efron(log_hz, exit_time, event, age)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_cindex(
        self,
        model: CoxMLP,
        features: Tensor,
        age: Tensor,
        exit_times: np.ndarray,
        events: np.ndarray,
    ) -> float:
        """Compute Harrell's C-index on the validation set."""
        from lifelines.utils import concordance_index

        model.eval()
        chunk = 4096
        parts: list[Tensor] = []
        with torch.no_grad():
            for i in range(0, len(features), chunk):
                parts.append(model(features[i : i + chunk], age[i : i + chunk]).cpu())
        log_hz = torch.cat(parts).numpy()

        # Higher log_hz → higher risk → shorter expected survival.
        # concordance_index expects higher score → longer survival, so pass -log_hz.
        return float(concordance_index(exit_times, -log_hz, events))
