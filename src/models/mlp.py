"""PyTorch Lightning MLP wrapped to implement BasePredictor.

The underlying ``models.model.Model`` is unchanged. This module adds:
- ``predict_proba``: converts log-softmax logits to probabilities.
- ``save`` / ``load``: thin wrappers over PyTorch checkpoint I/O.
- ``fit``: orchestrates the Lightning Trainer using values from ``param_tennis.Param``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.model import Model
from src.models.base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class MLPPredictor(BasePredictor):
    """Wraps the existing PyTorch Lightning MLP as a ``BasePredictor``.

    Parameters
    ----------
    learning_rate : float
        Adam learning rate (default matches original training: 1e-6).
    batch_size : int
        Mini-batch size for training.
    epochs : int
        Maximum training epochs (early stopping may terminate sooner).
    checkpoint_dir : str
        Directory where model checkpoints are saved during training.
    """

    def __init__(
        self,
        learning_rate: float = 1e-6,
        batch_size: int = 128,
        epochs: int = 100,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self._model: Optional[Model] = None

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the MLP.

        Parameters
        ----------
        X, y : training features and binary labels.
        X_val, y_val : optional validation split. If omitted, 20% of training
            data is used for validation.
        """
        if X_val is None or y_val is None:
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
        else:
            X_train, y_train = X, y

        train_ds = self._make_dataset(X_train, y_train)
        val_ds = self._make_dataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self._model = Model(learning_rate=self.learning_rate)

        early_stop = EarlyStopping(monitor="val_acc", min_delta=0.01, patience=4, mode="max")
        checkpoint_cb = ModelCheckpoint(
            dirpath=self.checkpoint_dir, save_top_k=1, monitor="val_loss"
        )

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=[early_stop, checkpoint_cb],
            enable_progress_bar=True,
            logger=False,  # caller adds loggers (wandb, mlflow) externally
        )
        trainer.fit(self._model, train_loader, val_loader)
        logger.info("MLP training complete. Best checkpoint: %s", checkpoint_cb.best_model_path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for each match.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 36)

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Column 0 → P(player 2 wins), Column 1 → P(player 1 wins).
        """
        if self._model is None:
            raise RuntimeError("Call fit() or load() before predict_proba().")

        self._model.eval()
        tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            log_probs = self._model(tensor)          # log-softmax output
        probs = torch.exp(log_probs).numpy()          # convert to probabilities
        return probs

    def save(self, path: str) -> None:
        """Save model weights to ``path`` (PyTorch state dict, .pt file)."""
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been trained.")
        torch.save(self._model.state_dict(), path)
        logger.info("Saved MLP weights to %s", path)

    @classmethod
    def load(cls, path: str) -> MLPPredictor:
        """Load an MLP predictor from a saved state dict.

        Parameters
        ----------
        path : str
            Path to a .pt file saved by ``save()``.
        """
        predictor = cls()
        predictor._model = Model()
        predictor._model.load_state_dict(torch.load(path, map_location="cpu"))
        predictor._model.eval()
        logger.info("Loaded MLP from %s", path)
        return predictor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        """Convert numpy arrays to a one-hot-labeled TensorDataset."""
        feats = torch.tensor(X, dtype=torch.float32)
        labels = F.one_hot(torch.tensor(y, dtype=torch.int64), num_classes=2)
        return TensorDataset(feats, labels)
