#!/usr/bin/env python3
"""HuBERT + mel preprocessing for Conan.

Single-process GPU preprocessing: one ONNX Runtime session, batched inputs,
and HDF5 outputs for Stage 1 content-extractor distillation.

Usage:
    python entry/preprocess.py -c configs/preprocess_hubert.yaml
"""

import json
import random
import time
from pathlib import Path
from typing import Iterable, Union

import h5py
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_SUFFIXES = {".wav", ".flac"}


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


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio.astype(np.float32)


def _compute_mel(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
) -> np.ndarray:
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
        n_frames = _hubert_frames(len(audio))
        if n_frames <= 0:
            raise ValueError(f"{path} is too short for HuBERT: {len(audio)} samples")
        mel = _compute_mel(audio, sample_rate, n_mels, n_fft, hop_size, win_size)
        samples.append({"path": path, "audio": audio, "mel": mel, "n_frames": n_frames})
    return samples


def _select_providers(preprocessing_cfg: dict):
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


def _run_hubert_batch(session: ort.InferenceSession, input_names: set[str], samples: list[dict]) -> np.ndarray:
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


class HubertPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.audio_cfg = config.get("audio", {})
        self.data_cfg = config.get("data", {})
        self.preprocessing_cfg = config.get("preprocessing", {})

    def run(self):
        wavs_dir = _resolve_path(self.data_cfg.get("wavs_dir", "data/libritts/wavs"))
        output_dir = _resolve_path(self.data_cfg.get("hubert_emb_dir", "data/libritts/hubert_embeddings"))
        onnx_path = _resolve_path(self.data_cfg.get("hubert_onnx", "weights/hubert_batch.onnx"))
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

        n_valid = min(int(self.data_cfg.get("n_valid", 150)), len(files) // 10)
        valid_seed = int(self.data_cfg.get("valid_seed", 1234))
        if n_valid:
            rng = random.Random(valid_seed)
            valid_set = set(rng.sample(files, n_valid))
            train_files = [p for p in files if p not in valid_set]
            valid_files = sorted(valid_set, key=lambda p: p.stat().st_size)
        else:
            train_files = files
            valid_files = []

        providers = _select_providers(self.preprocessing_cfg)
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_names = {i.name for i in session.get_inputs()}

        print(f"Audio files: {len(files)}", flush=True)
        print(f"Preprocess splits: train={len(train_files)}, valid={len(valid_files)}", flush=True)
        print(f"Validation seed: {valid_seed}", flush=True)
        print(f"ONNX model: {onnx_path}", flush=True)
        print(f"ONNX providers: {session.get_providers()}", flush=True)
        print(f"Max batch size: {int(self.preprocessing_cfg.get('max_batch_size', 8))}", flush=True)

        t0 = time.time()
        ok_train = self._write_split(train_h5, train_files, session, input_names, "preprocess-train")
        ok_valid = self._write_split(valid_h5, valid_files, session, input_names, "preprocess-valid")

        meta = {
            "total": ok_train + ok_valid,
            "train": ok_train,
            "valid": ok_valid,
            "wavs_dir": str(wavs_dir.relative_to(PROJECT_ROOT) if wavs_dir.is_relative_to(PROJECT_ROOT) else wavs_dir),
            "onnx_model": str(onnx_path.relative_to(PROJECT_ROOT) if onnx_path.is_relative_to(PROJECT_ROOT) else onnx_path),
            "providers": session.get_providers(),
            "max_batch_size": int(self.preprocessing_cfg.get("max_batch_size", 8)),
            "n_valid": n_valid,
            "valid_seed": valid_seed,
            "split": "random",
            "elapsed_sec": round(time.time() - t0, 3),
        }
        with open(output_dir / "conan_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\nDone: {json.dumps(meta, ensure_ascii=False)}", flush=True)

    def _write_split(
        self,
        h5_path: Path,
        files: list[Path],
        session: ort.InferenceSession,
        input_names: set[str],
        split_name: str,
    ) -> int:
        batch_size = int(self.preprocessing_cfg.get("max_batch_size", 8))
        if batch_size <= 0:
            raise ValueError("preprocessing.max_batch_size must be positive")

        ok = 0
        with h5py.File(h5_path, "w") as h5f:
            for key, value in {
                "sample_rate": int(self.audio_cfg.get("sample_rate", 16000)),
                "n_mels": int(self.audio_cfg.get("num_mels", 80)),
                "hop_size": int(self.audio_cfg.get("hop_size", 320)),
                "n_fft": int(self.audio_cfg.get("n_fft", 1024)),
                "win_size": int(self.audio_cfg.get("win_size", 1024)),
            }.items():
                h5f.attrs[key] = value

            pbar = tqdm(total=len(files), desc=split_name, unit="files", dynamic_ncols=True)
            start_time = time.time()
            for batch_files in _batches(files, batch_size):
                samples = _load_batch(batch_files, self.audio_cfg)
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
