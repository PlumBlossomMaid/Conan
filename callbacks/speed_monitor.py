"""SpeedMonitor — persistent training speed monitoring with VDL logging.

Logs timing breakdown to VDL under the ``perf/`` namespace, separate from
``train/`` and ``val/`` metrics, so performance curves are visible in a
dedicated VDL section.

Usage::

    from callbacks.speed_monitor import SpeedMonitor

    trainer = ocean.Trainer(
        ...,
        callbacks=[
            ConanProgressBar(),
            ModelCheckpoint(...),
            SpeedMonitor(window_size=100, verbose=True),
        ],
    )
"""

import time
from typing import Any

from ocean.callbacks.callback import Callback


class SpeedMonitor(Callback):
    """Monitor training speed and log timing breakdown to VDL.

    Args:
        window_size: Number of batches for rolling average (default: 100).
        verbose: If True, prints timing summary at epoch end (default: False).
    """

    def __init__(self, window_size: int = 100, verbose: bool = False) -> None:
        self.window_size = window_size
        self.verbose = verbose

        self._batch_start: float = 0.0
        self._forward_end: float = 0.0

        self._step_times: list[float] = []
        self._forward_times: list[float] = []
        self._backward_times: list[float] = []
        self._val_step_times: list[float] = []

        self._epoch_start: float = 0.0
        self._last_logged_step: int = -1

    def on_train_batch_start(self, trainer: Any, model: Any, batch: Any, batch_idx: int) -> None:
        self._batch_start = time.perf_counter()

    def on_before_backward(self, trainer: Any, model: Any, loss: Any) -> None:
        """Called after forward + loss compute, just before ``loss.backward()``."""
        self._forward_end = time.perf_counter()

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        now = time.perf_counter()
        step_time = now - self._batch_start
        forward_time = self._forward_end - self._batch_start
        backward_time = now - self._forward_end

        self._step_times.append(step_time)
        self._forward_times.append(forward_time)
        self._backward_times.append(backward_time)

        # Trim rolling window
        if len(self._step_times) > self.window_size:
            self._step_times.pop(0)
            self._forward_times.pop(0)
            self._backward_times.pop(0)

        opt_step = trainer.optimizer_step
        if opt_step == 0 or opt_step == self._last_logged_step:
            return
        self._last_logged_step = opt_step
        log_every = getattr(trainer, 'log_every_n_steps', 1)
        write_step = opt_step - 1
        if write_step % log_every == 0:
            trainer._logger_connector.log_metrics(
                {
                    "perf/step_time": sum(self._step_times) / len(self._step_times),
                    "perf/forward_time": sum(self._forward_times) / len(self._forward_times),
                    "perf/backward_time": sum(self._backward_times) / len(self._backward_times),
                },
                write_step,
            )

    def on_validation_batch_start(
        self, trainer: Any, model: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if batch_idx == 0:
            self._val_batch_start = time.perf_counter()

    def on_validation_batch_end(
        self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        elapsed = time.perf_counter() - self._val_batch_start
        self._val_step_times.append(elapsed)
        self._val_batch_start = time.perf_counter()
        if len(self._val_step_times) > self.window_size:
            self._val_step_times.pop(0)

    def on_validation_epoch_end(self, trainer: Any, model: Any) -> None:
        if self._val_step_times:
            opt_step = trainer.optimizer_step
            write_step = max(0, opt_step - 1)
            trainer._logger_connector.log_metrics(
                {
                    "perf/val_step_time": sum(self._val_step_times) / len(self._val_step_times),
                    "perf/val_batches": len(self._val_step_times),
                },
                write_step,
            )
            self._val_step_times.clear()

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        epoch_time = time.perf_counter() - self._epoch_start
        opt_step = trainer.optimizer_step
        if opt_step > 0:
            trainer._logger_connector.log_metrics(
                {
                    "perf/epoch_time": epoch_time,
                    "perf/samples_per_sec": (len(self._step_times) / epoch_time) if epoch_time > 0 else 0,
                },
                opt_step,
            )

        if self.verbose and self._step_times:
            avg_step = sum(self._step_times) / len(self._step_times)
            avg_fwd = sum(self._forward_times) / len(self._forward_times)
            avg_bwd = sum(self._backward_times) / len(self._backward_times)
            print(
                f"  [SpeedMonitor] epoch avg: {avg_step:.3f}s/it  "
                f"(fwd={avg_fwd:.3f}s  bwd={avg_bwd:.3f}s  "
                f"samples/s={len(self._step_times) / epoch_time:.1f})"
            )
