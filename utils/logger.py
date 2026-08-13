"""File-only logger — writes detailed training logs to {log_dir}/train.log.

Console output is intentionally left to tqdm only (via Ocean progress bar).
All detailed metrics, events, and errors go to:
  1. VDL (via trainer._logger_connector.log_metrics / model.log)
  2. Local log file (via this logger)

Usage:
    from utils.logger import get_logger

    logger = get_logger(config["log_dir"])
    logger.info("=" * 50)
    logger.info("Stage 1 — Training Start")
    logger.info(f"  Dataset: {len(train_dataset)} samples")
"""

import logging
from pathlib import Path


def get_logger(log_dir: str, filename: str = "train.log", level: int = logging.INFO) -> logging.Logger:
    """Create a file-only logger.

    Args:
        log_dir: Directory for the log file (created if missing).
        filename: Log file name (default: ``train.log``).
        level: Logging level (default: ``INFO``).

    Returns:
        A configured ``logging.Logger`` instance writing only to file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(str(log_path.resolve()))
    logger.setLevel(level)

    # Avoid duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    fh = logging.FileHandler(str(log_path / filename), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
