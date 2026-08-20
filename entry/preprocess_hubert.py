#!/usr/bin/env python3
"""HuBERT + mel preprocessing for Conan.

Uses the aligned Paddle HuBERT teacher checkpoint to create HDF5 distillation
targets for Stage 1 content-extractor training.

Usage:
    python entry/preprocess.py -c configs/preprocess_hubert.yaml
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Union

import h5py
import librosa
import numpy as np
import paddle
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.batching import batch_by_files
from layers.hubert import HubertTeacher

AUDIO_SUFFIXES = {".wav", ".flac"}


def _resolve_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _list_audio(src: Path) -> list[Path]:
    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]
    return sorted(files, key=lambda p: p.stat().st_size)


def _batches(
    files: list[Path],
    max_batch_size: int,
    max_batch_frames: int,
    sample_rate: int,
    hop_size: int,
) -> list[list[Path]]:
    if max_batch_frames <= 0:
        return batch_by_files(
            files,
            lambda path: 1,
            max_batch_frames=0,
            max_batch_size=max_batch_size,
            sort_by_len=False,
        )
    return batch_by_files(
        files,
        lambda path: _estimate_mel_frames(path, sample_rate, hop_size),
        max_batch_frames=max_batch_frames,
        max_batch_size=max_batch_size,
        sort_by_len=True,
        grid=1,
    )


def _estimate_mel_frames(path: Path, sample_rate: int, hop_size: int) -> int:
    info = sf.info(str(path))
    frames = int(info.frames)
    if info.samplerate != sample_rate:
        frames = int(np.ceil(frames * sample_rate / info.samplerate))
    return max(1, int(np.ceil(frames / hop_size)))


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


def _run_hubert_batch(model: HubertTeacher, samples: list[dict]) -> list[np.ndarray]:
    embeddings = []
    with paddle.no_grad():
        for sample in samples:
            source = paddle.to_tensor(sample["audio"].reshape([1, 1, -1]))
            embeddings.append(model(source).numpy()[0])
    return embeddings


class HubertPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.audio_cfg = config.get("audio", {})
        self.data_cfg = config.get("data", {})
        self.preprocessing_cfg = config.get("preprocessing", {})

    def run(self):
        wavs_dir = _resolve_path(self.data_cfg.get("wavs_dir", "data/libritts/wavs"))
        output_dir = _resolve_path(self.data_cfg.get("hubert_emb_dir", "data/libritts/hubert_embeddings"))
        checkpoint_path = _resolve_path(self.data_cfg.get("hubert_checkpoint", "weights/hubert.pdparams"))
        output_dir.mkdir(parents=True, exist_ok=True)

        if not wavs_dir.exists():
            raise FileNotFoundError(f"Waveform directory not found: {wavs_dir}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"HuBERT Paddle checkpoint not found: {checkpoint_path}")

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

        device = str(self.preprocessing_cfg.get("device", self.config.get("device", "gpu")))
        if device == "gpu":
            device = "gpu:0"
        paddle.set_device(device)
        model = HubertTeacher()
        model.load_pretrained(checkpoint_path)
        model.eval()

        max_batch_size = int(self.preprocessing_cfg.get("max_batch_size", 8))
        max_batch_frames = int(self.preprocessing_cfg.get("max_batch_frames", 0))

        print(f"Audio files: {len(files)}", flush=True)
        print(f"Preprocess splits: train={len(train_files)}, valid={len(valid_files)}", flush=True)
        print(f"Validation seed: {valid_seed}", flush=True)
        print(f"HuBERT checkpoint: {checkpoint_path}", flush=True)
        print(f"Paddle device: {paddle.get_device()}", flush=True)
        print(f"Max batch size: {max_batch_size}", flush=True)
        print(f"Max batch frames: {max_batch_frames if max_batch_frames > 0 else 'disabled'}", flush=True)

        t0 = time.time()
        ok_train = self._write_split(train_h5, train_files, model, "preprocess-train")
        ok_valid = self._write_split(valid_h5, valid_files, model, "preprocess-valid")

        meta = {
            "total": ok_train + ok_valid,
            "train": ok_train,
            "valid": ok_valid,
            "wavs_dir": str(wavs_dir.relative_to(PROJECT_ROOT) if wavs_dir.is_relative_to(PROJECT_ROOT) else wavs_dir),
            "hubert_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT) if checkpoint_path.is_relative_to(PROJECT_ROOT) else checkpoint_path),
            "paddle_device": paddle.get_device(),
            "max_batch_size": max_batch_size,
            "max_batch_frames": max_batch_frames,
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
        model: HubertTeacher,
        split_name: str,
    ) -> int:
        batch_size = int(self.preprocessing_cfg.get("max_batch_size", 8))
        max_batch_frames = int(self.preprocessing_cfg.get("max_batch_frames", 0))
        sample_rate = int(self.audio_cfg.get("sample_rate", 16000))
        hop_size = int(self.audio_cfg.get("hop_size", 320))
        batches = _batches(files, batch_size, max_batch_frames, sample_rate, hop_size)

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
            for batch_files in batches:
                samples = _load_batch(batch_files, self.audio_cfg)
                emb = _run_hubert_batch(model, samples)
                for i, sample in enumerate(samples):
                    n_frames = min(sample["n_frames"] + 1, emb[i].shape[0])
                    grp = h5f.create_group(f"{ok:08d}")
                    grp.create_dataset("mel", data=sample["mel"], compression="gzip")
                    grp.create_dataset("hubert", data=emb[i][:n_frames].astype(np.float32), compression="gzip")
                    grp.attrs["source_path"] = str(sample["path"].relative_to(PROJECT_ROOT))
                    ok += 1

                pbar.update(len(batch_files))
                elapsed = time.time() - start_time
                if elapsed > 0:
                    pbar.set_postfix(speed=f"{ok / elapsed:.1f}/s")
            pbar.close()

        return ok
