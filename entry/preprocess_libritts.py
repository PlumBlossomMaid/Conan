"""Validate audio prepared by an external dataset tool.

Usage:
    python entry/preprocess.py -c configs/preprocess_libritts.yaml

Conan consumes an already prepared flat audio directory. Downloading,
extracting, and flattening datasets belong to the dataset tool, not Conan.
"""

from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_SUFFIXES = {".wav", ".flac"}


def _resolve_path(path: Union[str, Path]) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


class LibriTTSPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config.get("data", {})

    def run(self):
        audio_dir = _resolve_path(self.data_cfg.get("audio_dir", "data/libritts/wavs"))
        if not audio_dir.exists():
            raise FileNotFoundError(f"Prepared audio directory not found: {audio_dir}")

        audio_files = sorted(
            path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )
        if not audio_files:
            raise RuntimeError(f"No prepared audio files found in {audio_dir}")

        total_size = sum(path.stat().st_size for path in audio_files)
        result = {"audio_dir": str(audio_dir), "audio_files": len(audio_files), "total_bytes": total_size}
        print(f"Prepared audio: {result}", flush=True)
        return len(audio_files), total_size
