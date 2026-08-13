"""Dataset construction shared by the Stage 2 and Stage 3 models.

Both stages read waveforms through :class:`~layers.dataset.ConanDataset` with
the same ``audio``/``data`` config sections, so the wiring lives here instead of
being duplicated per model.
"""

from layers.dataset import ConanDataset

from .training_utils import build_train_dataloader, build_val_dataloader


def build_conan_dataset(config: dict, is_train: bool, max_samples=None) -> ConanDataset:
    """Instantiate a ``ConanDataset`` from the config's audio/data sections.

    Args:
        config: Full training configuration dict.
        is_train: Enables training-time cropping and augmentation.
        max_samples: Cap on dataset size (used to keep validation quick).

    Returns:
        Configured ConanDataset.
    """
    audio_cfg = config.get("audio", {})
    data_cfg = config.get("data", {})

    kwargs = dict(
        data_dir=data_cfg["data_dir"],
        sample_rate=audio_cfg.get("sample_rate", 16000),
        n_mels=audio_cfg.get("num_mels", 80),
        n_fft=audio_cfg.get("n_fft", 1024),
        hop_size=audio_cfg.get("hop_size", 320),
        win_size=audio_cfg.get("win_size", 1024),
        is_train=is_train,
    )
    if max_samples is not None:
        kwargs["max_samples"] = max_samples
    return ConanDataset(**kwargs)


def build_conan_train_dataloader(config: dict):
    """Frame-budget training dataloader over the waveform dataset.

    Args:
        config: Full training configuration dict.

    Returns:
        Training DataLoader.
    """
    return build_train_dataloader(build_conan_dataset(config, is_train=True), config)


def build_conan_val_dataloader(config: dict):
    """Validation dataloader, capped at ``data.val_max_samples`` utterances.

    Args:
        config: Full training configuration dict.

    Returns:
        Validation DataLoader with batch size 1.
    """
    max_samples = config.get("data", {}).get("val_max_samples", 20)
    return build_val_dataloader(build_conan_dataset(config, is_train=False, max_samples=max_samples))
