"""Prepare LibriTTS data for Conan training.

Usage:
    python scripts/prepare_libritts.py -c configs/binarize.yaml

Steps:
    1. Read LibriTTS tarballs from ``data.archive_dir``
    2. Extract only audio files directly into ``data.wavs_dir``
    3. Flatten speaker/chapter paths into stable filenames
"""

import argparse
import shutil
import tarfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_SUFFIXES = {".wav", ".flac"}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parents = config.pop("base_config", None)
    if not parents:
        return config
    if isinstance(parents, str):
        parents = [parents]

    merged = {}
    for parent in parents:
        merged = _deep_update(merged, load_config(parent))
    return _deep_update(merged, config)


def _deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _flat_name(member_name: str) -> str:
    path = Path(member_name)
    stem = "_".join(path.with_suffix("").parts)
    return f"{stem}{path.suffix.lower()}"


def extract_audio_flat(tgz_path: Path, wavs_dir: Path) -> int:
    count = 0
    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if Path(member.name).suffix.lower() not in AUDIO_SUFFIXES:
                continue

            out_path = wavs_dir / _flat_name(member.name)
            if out_path.exists() and out_path.stat().st_size == member.size:
                count += 1
                continue

            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            count += 1
    return count


def flatten_existing_audio(src_root: Path, wavs_dir: Path) -> int:
    count = 0
    for src in sorted(src_root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        out_path = wavs_dir / _flat_name(str(src.relative_to(src_root)))
        if out_path.exists() and out_path.stat().st_size == src.stat().st_size:
            count += 1
            continue
        shutil.copy2(src, out_path)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare LibriTTS for Conan")
    parser.add_argument("-c", "--config", required=True, help="Path to binarization config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    preprocessing_cfg = config.get("preprocessing", {})

    archive_dir = _resolve_path(data_cfg.get("archive_dir", "data"))
    wavs_dir = _resolve_path(data_cfg.get("wavs_dir", "data/libritts/wavs"))
    wavs_dir.mkdir(parents=True, exist_ok=True)

    if preprocessing_cfg.get("flatten_existing", False):
        print(f"Flattening extracted audio from {archive_dir} to {wavs_dir} ...", flush=True)
        count = flatten_existing_audio(archive_dir, wavs_dir)
    else:
        tgz_files = sorted(archive_dir.glob("*.tar.gz"))
        if not tgz_files:
            raise FileNotFoundError(f"No .tar.gz files found in {archive_dir}")
        print(f"Extracting {len(tgz_files)} tarballs directly to {wavs_dir} ...", flush=True)
        count = 0
        for tgz in tgz_files:
            n = extract_audio_flat(tgz, wavs_dir)
            count += n
            print(f"  {tgz.name}: {n} audio files", flush=True)

    total_size = sum(f.stat().st_size for f in wavs_dir.iterdir() if f.is_file())
    print(f"\n=== LibriTTS ready at {wavs_dir} ===", flush=True)
    print(f"  {count} files, {total_size / 1024**3:.1f} GB", flush=True)
    print("\nNext step:", flush=True)
    print(f"  python scripts/binarize_conan.py -c {args.config}", flush=True)


if __name__ == "__main__":
    main()
