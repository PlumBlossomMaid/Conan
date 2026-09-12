"""Unified training entry point for all Conan stages.

The stage is selected entirely by the config file's ``task_cls``, so every stage
shares one command:

    python entry/train.py -c configs/content_extractor.yaml
    python entry/train.py -c configs/vocoder.yaml
    python entry/train.py -c configs/main.yaml

``-c`` is the only accepted argument; everything else lives in the config.
"""

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ocean
import yaml
from ocean.callbacks import ModelCheckpoint

from callbacks.file_metrics import FileMetricsCallback
from callbacks.speed_monitor import SpeedMonitor
from callbacks.progress_bar import ConanProgressBar
from utils.config_utils import apply_overrides, load_config
from utils.dotdict import DotDict
from utils.model_utils import freeze_params, print_model_summary


def get_task_class(task_cls: str):
    """Resolve a ``task_cls`` string such as ``models.vocoder.VocoderModel``.

    Args:
        task_cls: Fully qualified class path.

    Returns:
        The model class.
    """
    module_path, _, class_name = task_cls.rpartition(".")
    if not module_path:
        raise ValueError(f"task_cls must be a full dotted path, got: {task_cls!r}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_logger(config: dict, log_dir: Path):
    """Create the experiment logger named by ``logger`` in the config.

    ``log_version`` pins the VisualDL ``version_N`` directory so a run that
    restarts from a checkpoint keeps writing into the same directory and the
    VDL curves stay continuous instead of splitting into a new version.
    """
    name = str(config.get("logger", "tensorboard")).lower()
    if name == "visualdl":
        version = config.get("log_version")
        return ocean.loggers.VisualDLLogger(
            str(log_dir), version=int(version) if version is not None else None
        )
    return ocean.loggers.TensorBoardLogger(str(log_dir))


def main():
    parser = argparse.ArgumentParser(description="Train a Conan stage.")
    parser.add_argument("-c", "--config", required=True, help="Path to the stage config YAML")
    parser.add_argument(
        "-o", "--override", action="append", default=None,
        metavar="KEY=VALUE",
        help="Override a config value (repeatable, YAML-typed). "
             "Example: -o data.hdf5_path=/tmp/train.h5 -o training.batch_size=16",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    apply_overrides(config, args.override)
    if "task_cls" not in config:
        raise ValueError(f"{args.config} must define 'task_cls'")
    DotDict(config).print_dict()

    train_cfg = config.get("training", {})
    work_dir = PROJECT_ROOT / config.get("work_dir", "ckpts")
    log_dir = Path(config.get("log_dir", work_dir / "logs"))
    ckpt_dir = Path(config.get("ckpt_dir", work_dir / "ckpt"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    task_cls = get_task_class(config["task_cls"])
    print(f"Task: {config['task_cls']}")
    model = task_cls(config)

    frozen = config.get("pretrained", {}).get("frozen_params", [])
    if frozen:
        num_frozen = freeze_params(model, frozen)
        print(f"Froze {num_frozen} parameter tensors matching {frozen}")
    print_model_summary(model)

    train_loader = model.train_dataloader()
    val_loader = model.val_dataloader()

    monitor = train_cfg.get("monitor", "val/loss")
    val_check_batches = train_cfg.get("val_check_interval", 1000) * train_cfg.get(
        "accumulate_grad_batches", 1
    )
    callbacks = [
        ConanProgressBar(),
        SpeedMonitor(window_size=100, verbose=True),
        FileMetricsCallback(log_dir),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            save_last=True,
            save_top_k=train_cfg.get("num_ckpt_keep", 3),
            monitor=monitor,
            mode=train_cfg.get("monitor_mode", "min"),
        ),
    ]

    last_ckpt = ckpt_dir / "last.pdparams"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() else None
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")

    trainer = ocean.Trainer(
        max_steps=train_cfg.get("steps", train_cfg.get("max_steps", -1)),
        accelerator=train_cfg.get("accelerator", "gpu"),
        devices=train_cfg.get("devices", 1),
        precision=train_cfg.get("precision", "32-true"),
        gradient_clip_val=train_cfg.get("clip_grad", 0.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=train_cfg.get("log_every", 100),
        val_check_interval=val_check_batches,
        num_sanity_val_steps=train_cfg.get("num_sanity_val_steps", 1),
        default_root_dir=str(work_dir),
        callbacks=callbacks,
        logger=build_logger(config, log_dir),
    )

    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )
    except Exception:
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
