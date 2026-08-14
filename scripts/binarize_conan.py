#!/usr/bin/env python3
"""Conan HuBERT + mel preprocessing.

Single-process GPU preprocessing: one ONNX Runtime session, batched inputs,
and HDF5 outputs for Stage 1 content-extractor training.

Usage:
    python scripts/binarize_conan.py -c configs/binarize.yaml
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable, Optional, Union

import h5py
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_SUFFIXES = {".wav", ".flac"}


def load_config(config_path: str) -> dict:
    """Load YAML config and recursively merge ``base_config`` parents."""
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


def _resolve_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _list_audio(src: Path) -> list[Path]:
    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]
    return sorted(files, key=lambda p: p.stat().st_size)


def _batches(files: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(files), batch_size):
        yield files[start:start + batch_size]


def _hubert_frames(n_samples: int) -> int:
    if n_samples < 400:
        return 0
    return (n_samples - 640) // 320 + 1


def _load_audio(path: Path, sample_rate: int) -> Optional[np.ndarray]:
    import soundfile as sf

    try:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio.astype(np.float32)
    except Exception as exc:
        print(f"\n[SKIP] {path.name}: {exc}", flush=True)
        return None


def _compute_mel(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
) -> np.ndarray:
    import librosa

    spec = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hann",
        center=True,
    )
    mag = np.abs(spec)
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel = np.dot(mel_basis, mag)
    return np.log10(np.clip(mel, 1e-5, None)).astype(np.float32)


def _load_batch(batch_files: list[Path], audio_cfg: dict):
    samples = []
    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    n_mels = int(audio_cfg.get("num_mels", 80))
    n_fft = int(audio_cfg.get("n_fft", 1024))
    hop_size = int(audio_cfg.get("hop_size", 320))
    win_size = int(audio_cfg.get("win_size", 1024))

    for path in batch_files:
        audio = _load_audio(path, sample_rate)
        if audio is None:
            continue
        n_frames = _hubert_frames(len(audio))
        if n_frames <= 0:
            print(f"\n[SKIP] {path.name}: too short for HuBERT", flush=True)
            continue
        mel = _compute_mel(audio, sample_rate, n_mels, n_fft, hop_size, win_size)
        samples.append({"path": path, "audio": audio, "mel": mel, "n_frames": n_frames})
    return samples


def _select_providers(ort, preprocessing_cfg: dict):
    provider = str(preprocessing_cfg.get("onnx_provider", "auto"))
    device_id = int(preprocessing_cfg.get("device_id", 0))
    allow_cpu = bool(preprocessing_cfg.get("allow_cpu_fallback", False))
    available = ort.get_available_providers()

    provider_map = {
        "ROCMExecutionProvider": ("ROCMExecutionProvider", {"device_id": device_id}),
        "CUDAExecutionProvider": ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider": "CPUExecutionProvider",
    }

    if provider != "auto":
        if provider not in available:
            raise RuntimeError(f"Requested ONNX provider {provider!r} not available; available={available}")
        return [provider_map[provider]]

    for candidate in ("ROCMExecutionProvider", "CUDAExecutionProvider"):
        if candidate in available:
            return [provider_map[candidate]]
    if allow_cpu and "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    raise RuntimeError(f"No GPU ONNX provider available; available={available}")


def _run_hubert_batch(session, input_names: set[str], samples: list[dict]) -> np.ndarray:
    max_len = max(len(s["audio"]) for s in samples)
    max_frames = max(s["n_frames"] for s in samples)
    source = np.zeros([len(samples), 1, max_len], dtype=np.float32)
    mask = np.ones([len(samples), max_frames + 1], dtype=np.float32)

    for i, sample in enumerate(samples):
        audio = sample["audio"]
        n_frames = sample["n_frames"]
        source[i, 0, :len(audio)] = audio
        mask[i, :n_frames + 1] = 0.0

    feed = {"source": source}
    if "key_padding_mask" in input_names:
        feed["key_padding_mask"] = mask
    return session.run(None, feed)[0]


def _write_split(h5_path: Path, files: list[Path], session, input_names: set[str], config: dict, split_name: str) -> int:
    audio_cfg = config.get("audio", {})
    preprocessing_cfg = config.get("preprocessing", {})
    batch_size = int(preprocessing_cfg.get("max_batch_size", 8))
    if batch_size <= 0:
        raise ValueError("preprocessing.max_batch_size must be positive")

    ok = 0
    with h5py.File(h5_path, "w") as h5f:
        for key, value in {
            "sample_rate": int(audio_cfg.get("sample_rate", 16000)),
            "n_mels": int(audio_cfg.get("num_mels", 80)),
            "hop_size": int(audio_cfg.get("hop_size", 320)),
            "n_fft": int(audio_cfg.get("n_fft", 1024)),
            "win_size": int(audio_cfg.get("win_size", 1024)),
        }.items():
            h5f.attrs[key] = value

        pbar = tqdm(total=len(files), desc=split_name, unit="files", dynamic_ncols=True)
        start_time = time.time()
        for batch_files in _batches(files, batch_size):
            samples = _load_batch(batch_files, audio_cfg)
            if not samples:
                pbar.update(len(batch_files))
                continue

            emb = _run_hubert_batch(session, input_names, samples)
            for i, sample in enumerate(samples):
                n_frames = min(sample["n_frames"] + 1, emb.shape[1])
                grp = h5f.create_group(f"{ok:08d}")
                grp.create_dataset("mel", data=sample["mel"], compression="gzip")
                grp.create_dataset("hubert", data=emb[i, :n_frames].astype(np.float32), compression="gzip")
                grp.attrs["source_path"] = str(sample["path"].relative_to(PROJECT_ROOT))
                ok += 1

            pbar.update(len(batch_files))
            elapsed = time.time() - start_time
            if elapsed > 0:
                pbar.set_postfix(speed=f"{ok / elapsed:.1f}/s")
        pbar.close()

    return ok


def main():
    parser = argparse.ArgumentParser(description="Binarize Conan training data")
    parser.add_argument("-c", "--config", required=True, help="Path to binarization config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    preprocessing_cfg = config.get("preprocessing", {})

    wavs_dir = _resolve_path(data_cfg.get("wavs_dir", "data/libritts/wavs"))
    output_dir = _resolve_path(data_cfg.get("hubert_emb_dir", "data/libritts/hubert_embeddings"))
    onnx_path = _resolve_path(data_cfg.get("hubert_onnx", "weights/hubert_batch.onnx"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not wavs_dir.exists():
        raise FileNotFoundError(f"Waveform directory not found: {wavs_dir}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"HuBERT ONNX model not found: {onnx_path}")

    files = _list_audio(wavs_dir)
    if not files:
        raise RuntimeError(f"No audio files found in {wavs_dir}")

    train_h5 = output_dir / "train.h5"
    valid_h5 = output_dir / "valid.h5"
    for h5_path in (train_h5, valid_h5):
        if h5_path.exists():
            raise FileExistsError(f"{h5_path} exists. Delete it manually before re-running.")

    n_valid = min(int(data_cfg.get("n_valid", 150)), len(files) // 10)
    valid_seed = int(data_cfg.get("valid_seed", 1234))
    if n_valid:
        rng = random.Random(valid_seed)
        valid_set = set(rng.sample(files, n_valid))
        train_files = [p for p in files if p not in valid_set]
        valid_files = sorted(valid_set, key=lambda p: p.stat().st_size)
    else:
        train_files = files
        valid_files = []

    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    providers = _select_providers(ort, preprocessing_cfg)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = {i.name for i in session.get_inputs()}

    print(f"Audio files: {len(files)}", flush=True)
    print(f"Train/valid: {len(train_files)}/{len(valid_files)}", flush=True)
    print(f"Validation seed: {valid_seed}", flush=True)
    print(f"ONNX model: {onnx_path}", flush=True)
    print(f"ONNX providers: {session.get_providers()}", flush=True)
    print(f"Max batch size: {int(preprocessing_cfg.get('max_batch_size', 8))}", flush=True)

    t0 = time.time()
    ok_train = _write_split(train_h5, train_files, session, input_names, config, "train")
    ok_valid = _write_split(valid_h5, valid_files, session, input_names, config, "valid")

    meta = {
        "total": ok_train + ok_valid,
        "train": ok_train,
        "valid": ok_valid,
        "wavs_dir": str(wavs_dir.relative_to(PROJECT_ROOT) if wavs_dir.is_relative_to(PROJECT_ROOT) else wavs_dir),
        "onnx_model": str(onnx_path.relative_to(PROJECT_ROOT) if onnx_path.is_relative_to(PROJECT_ROOT) else onnx_path),
        "providers": session.get_providers(),
        "max_batch_size": int(preprocessing_cfg.get("max_batch_size", 8)),
        "n_valid": n_valid,
        "valid_seed": valid_seed,
        "split": "random",
        "elapsed_sec": round(time.time() - t0, 3),
    }
    with open(output_dir / "conan_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nDone: {json.dumps(meta, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
