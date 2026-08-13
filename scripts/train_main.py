"""Train Conan Stage 2: Main Model (ASE + Pitch Predictor + Mel Decoder).

Usage:
    python scripts/train_main.py --config configs/main.yaml [--resume]

Requires a pretrained content extractor checkpoint.
Trains with manual G/D alternating optimization.
160k steps per paper.
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

from models.conan_main import ConanMainModel
from layers.stream_content_extractor import StreamContentExtractor
from layers.dataset import ConanDataset


def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_content_extractor(ckpt_path: str, config: dict) -> StreamContentExtractor:
    """Load pretrained StreamContentExtractor from checkpoint."""
    sce_cfg = config.get("content_extractor", {})
    audio_cfg = config.get("audio", {})

    extractor = StreamContentExtractor(
        input_dim=audio_cfg.get("num_mels", 80),
        d_model=sce_cfg.get("d_model", 512),
        nhead=sce_cfg.get("nhead", 8),
        num_layers=sce_cfg.get("num_layers", 6),
        output_dim=sce_cfg.get("output_dim", 256),
        chunk_size=sce_cfg.get("chunk_size", 4),
        left_context=sce_cfg.get("left_context", 1),
        right_context=sce_cfg.get("right_context", 2),
    )

    ckpt = paddle.load(ckpt_path)
    if "state_dict" in ckpt:
        sd = {k.replace("extractor.", ""): v
              for k, v in ckpt["state_dict"].items()
              if k.startswith("extractor.")}
        extractor.set_state_dict(sd)
    else:
        extractor.set_state_dict(ckpt)

    for p in extractor.parameters():
        p.stop_gradient = True
    extractor.eval()
    print(f"  Loaded content extractor from: {ckpt_path}")
    return extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/main.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--content-ckpt", type=str, default=None,
                        help="Override content extractor checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    audio_cfg = config["audio"]
    data_cfg = config["data"]

    # ── Load pretrained content extractor ──
    content_ckpt = args.content_ckpt or data_cfg.get("ckpt_content_extractor")
    if content_ckpt and os.path.exists(content_ckpt):
        content_extractor = load_content_extractor(content_ckpt, config)
    else:
        print("  WARNING: No content extractor checkpoint found. Using mel as content.")
        content_extractor = None

    # ── Dataset ──
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
        max_samples=20,
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
    model = ConanMainModel(config, content_extractor=content_extractor)
    gen_params = sum(p.numel() for p in model.generator_params())
    disc_params = sum(p.numel() for p in model.mel_disc.parameters())
    print(f"  Generator params: {gen_params:,}")
    print(f"  Discriminator params: {disc_params:,}")

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
        limit_val_batches=2,
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
