"""Train Conan Stage 3: Causal Shuffle Vocoder.

Usage:
    python scripts/train_vocoder.py --config configs/vocoder.yaml [--resume]

Trains the fully causal vocoder with pixel-shuffle upsampling.
600k steps per paper, manual G/D optimization.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import paddle
import ocean
from ocean.callbacks import ModelCheckpoint, LearningRateMonitor
from paddle.io import DataLoader

from models.vocoder import VocoderModel
from layers.dataset import ConanDataset


def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/vocoder.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    audio_cfg = config["audio"]
    data_cfg = config["data"]

    # ── Dataset (audio + mel pairs) ──
    train_dataset = ConanDataset(
        data_dir=data_cfg["data_dir"],
        sample_rate=audio_cfg["sample_rate"],
        n_mels=audio_cfg["num_mels"],
        n_fft=audio_cfg["n_fft"],
        hop_size=audio_cfg["hop_size"],
        win_size=audio_cfg["win_size"],
        is_train=True,
    )
    valid_dataset = ConanDataset(
        data_dir=data_cfg["data_dir"],
        sample_rate=audio_cfg["sample_rate"],
        n_mels=audio_cfg["num_mels"],
        n_fft=audio_cfg["n_fft"],
        hop_size=audio_cfg["hop_size"],
        win_size=audio_cfg["win_size"],
        is_train=False,
        max_samples=10,
    )

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
    model = VocoderModel(config)
    print(f"  Vocoder params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Logger ──
    logger = ocean.loggers.TensorBoardLogger(config["log_dir"])

    # ── Trainer ──
    ckpt_dir = config["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = None
    if args.resume:
        import glob, re
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.ckpt")),
                       key=lambda p: int(re.search(r"step[=_ ]?(\d+)", os.path.basename(p)).group(1)))
        if ckpts:
            ckpt_path = ckpts[-1]
            print(f"  Resuming from: {ckpt_path}")

    trainer = ocean.Trainer(
        max_steps=train_cfg["steps"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        precision=train_cfg["precision"],
        gradient_clip_val=train_cfg["clip_grad"],
        log_every_n_steps=train_cfg["log_every"],
        val_check_interval=train_cfg["val_check_interval"],
        limit_val_batches=1,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_dir, save_last=True,
                            every_n_train_steps=train_cfg["val_check_interval"]),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
