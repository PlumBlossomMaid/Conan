"""FileMetricsCallback — writes training/validation metrics to train.log.

Dumps a structured summary line at the same frequency as VDL writes
(``log_every_n_steps``), so you can review training history without
opening VDL.

The log file is created by ``get_logger()`` from ``utils.logger``.
"""

import time
from typing import Any

from ocean.callbacks.callback import Callback
from utils.logger import get_logger


class FileMetricsCallback(Callback):
    """Write periodic training metrics to a local log file.

    Args:
        log_dir: Directory for the log file (same as ``config["log_dir"]``).
    """

    def __init__(self, log_dir: str) -> None:
        self.logger = get_logger(log_dir)
        self._train_start: float = 0.0

    # ── Event logging ──

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        self.logger.info("=" * 60)
        self.logger.info("Training started")

    def on_fit_end(self, trainer: Any, model: Any) -> None:
        self.logger.info(f"Training finished — total steps: {trainer.dataloader_step}")

    # ── Periodic metric logging ──

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        pass

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        epoch = trainer.current_epoch
        opt_step = trainer.optimizer_step
        self.logger.info(f"Epoch {epoch} done — optimizer_step={opt_step}")

    # ── Error ──

    def on_exception(self, trainer: Any, model: Any, exception: BaseException) -> None:
        self.logger.exception("Training crashed")
