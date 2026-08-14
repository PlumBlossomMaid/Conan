"""Train Conan Stage 1: Stream Content Extractor.

Usage:
    python scripts/train_content_extractor.py --config configs/content_extractor.yaml

Trains an Emformer encoder to distill HuBERT content labels via
cross-entropy loss. 80k steps per paper.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
import paddle
import ocean
from ocean.callbacks import ModelCheckpoint, TQDMProgressBar
from paddle.io import DataLoader

from models.content_extractor import ContentExtractorModel
from layers.dataset import ContentExtractorDataset
from callbacks.speed_monitor import SpeedMonitor
from callbacks.file_metrics import FileMetricsCallback
from utils.logger import get_logger


class ConanProgressBar(TQDMProgressBar):
    """Progress bar that shows dataloader step as string (avoids scientific notation)."""

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items["steps"] = str(trainer.dataloader_step)
        for k, v in items.items():
            if isinstance(v, float) and np.isnan(v):
                items[k] = "nan"
        return items


def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/content_extractor.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    audio_cfg = config["audio"]
    data_cfg = config["data"]

    # Save merged config so experiment directory is self-documenting.
    os.makedirs(config["log_dir"], exist_ok=True)
    with open(os.path.join(config["log_dir"], "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, encoding="utf-8")

    # ── File logger (detailed logs to train.log, console stays tqdm-only) ──
    log = get_logger(config["log_dir"])
    log.info("=" * 60)
    log.info(f"Content Extractor Stage 1 — config: {args.config}")
    if "resume" in config and config["resume"]:
        log.info(f"  Mode: resume")
    else:
        log.info(f"  Mode: fresh start")

    # ── Dataset ──
    train_dataset = ContentExtractorDataset(
        hdf5_path=data_cfg.get("hdf5_path", "data/libritts/hubert_embeddings/train.h5"),
        max_frames=audio_cfg.get("max_frames", 500),
    )
    valid_dataset = ContentExtractorDataset(
        hdf5_path=data_cfg.get("hdf5_path", "data/libritts/hubert_embeddings/train.h5"),
        max_frames=audio_cfg.get("max_frames", 500),
        max_samples=data_cfg["val_max_samples"],
    )
    log.info(f"  Train samples: {len(train_dataset)}")
    log.info(f"  Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collater,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=valid_dataset.collater,
    )

    # ── Model ──
    model = ContentExtractorModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Content Extractor params: {n_params:,}")
    log.info(f"  Model params: {n_params:,}")

    # ── Logger (visualdl or tensorboard) ──
    logger_type = config.get("logger", "tensorboard")
    if logger_type == "tensorboard":
        logger = ocean.loggers.TensorBoardLogger(config["log_dir"])
    else:
        logger = ocean.loggers.VisualDLLogger(config["log_dir"], version="latest")

    # ── Trainer ──
    ckpt_dir = config["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    last_pd = os.path.join(ckpt_dir, "last.pdparams")
    ckpt_path = last_pd if os.path.exists(last_pd) else None
    if ckpt_path:
        print(f"  Auto-resuming from: {ckpt_path}")
    # Convert optimizer-step interval to batch interval for Ocean's batch-based val check
    val_check_batches = train_cfg["val_check_interval"] * train_cfg["accumulate_grad_batches"]

    trainer = ocean.Trainer(
        max_steps=train_cfg["steps"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        precision=train_cfg["precision"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        gradient_clip_val=train_cfg["clip_grad"],
        log_every_n_steps=train_cfg["log_every"],
        val_check_interval=val_check_batches,
        num_sanity_val_steps=train_cfg["num_sanity_val_steps"],
        callbacks=[
            ConanProgressBar(),
            SpeedMonitor(window_size=100, verbose=True),
            FileMetricsCallback(config["log_dir"]),
            ModelCheckpoint(dirpath=ckpt_dir, save_last=True, save_top_k=train_cfg["num_ckpt_keep"],
                            monitor="val/loss", mode="min"),
        ],
        logger=logger,
    )

    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                    ckpt_path=ckpt_path)
    except Exception:
        log.exception("Training crashed")
        raise


if __name__ == "__main__":
    main()
