#!/usr/bin/env python3
"""HuBERT + mel preprocessing for Conan.

Uses the aligned Paddle HuBERT teacher checkpoint to create HDF5 distillation
targets for Stage 1 content-extractor training.

Usage:
    python entry/preprocess.py -c configs/preprocess_hubert.yaml
"""

import gc
import json
import random
import sys
import time
from concurrent.futures import Executor, ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Union

import h5py
import librosa
import numpy as np
import paddle
import soundfile as sf
from ppAudio.features import STFT
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.batching import batch_by_files
from layers.hubert import HubertTeacher, hubert_frame_count

AUDIO_SUFFIXES = {".wav", ".flac"}

# Per-batch cost model (ms), fit on the target hardware: HuBERT attention
# dominates once the padded window is non-trivial, with a fixed floor for
# launching the CNN + 9 transformer layers. Audio load / mel / HDF5 write run
# while the GPU is busy and only bind when the batch is tiny.
BATCH_COST_FLOOR_MS = 100.0
BATCH_COST_PER_TOKEN = 2.0e-6
BATCH_PIPELINE_BASE_MS = 8.0
BATCH_PIPELINE_PER_SAMPLE = 1.0


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
    max_attention_tokens: int,
    sample_rate: int,
    hop_size: int,
) -> list[list[Path]]:
    return batch_by_files(
        files,
        lambda path: _estimate_mel_frames(path, sample_rate, hop_size),
        max_batch_frames=max_batch_frames,
        max_batch_size=max_batch_size,
        max_attention_tokens=max_attention_tokens,
        sort_by_len=True,
        grid=1,
    )


def _estimated_batch_cost_ms(
    files: list[Path], sample_rate: int, hop_size: int
) -> float:
    """Cheap estimate of one batch's wall time in milliseconds.

    On this hardware HuBERT inference cost is dominated by the padded
    attention window, which scales with ``batch_size * max_frames^2``, with a
    large fixed floor for launching the CNN feature extractor and 9
    transformer layers. The fixed cost (audio load + mel + HDF5 write) runs
    while the GPU is busy, so it only matters when the batch is tiny.
    """
    max_frames = max(_estimate_mel_frames(path, sample_rate, hop_size) for path in files)
    return BATCH_COST_FLOOR_MS + BATCH_COST_PER_TOKEN * len(files) * max_frames * max_frames


def _total_wall_ms(
    files: list[Path],
    batch_size: int,
    max_batch_frames: int,
    max_attention_tokens: int,
    sample_rate: int,
    hop_size: int,
) -> float:
    """Estimated wall time (ms) to process ``files`` under these limits."""
    batches = _batches(
        files,
        batch_size,
        max_batch_frames,
        max_attention_tokens,
        sample_rate,
        hop_size,
    )
    total = 0.0
    for batch in batches:
        total += max(
            _estimated_batch_cost_ms(batch, sample_rate, hop_size),
            _estimated_pipeline_ms(batch, sample_rate, hop_size),
        )
    return total


def _estimated_pipeline_ms(
    files: list[Path], sample_rate: int, hop_size: int
) -> float:
    """Audio-load + mel-prep + HDF5-write time, overlapped with the GPU."""
    samples = sum(max(_estimate_mel_frames(path, sample_rate, hop_size), 1) for path in files)
    return BATCH_PIPELINE_BASE_MS + BATCH_PIPELINE_PER_SAMPLE * samples


def _tune_batch_limits(
    files: list[Path],
    max_batch_size: int,
    max_batch_frames: int,
    max_attention_tokens: int,
    sample_rate: int,
    hop_size: int,
    preprocessing_cfg: dict,
) -> tuple[int, int, int]:
    """Auto-tune batch limits from the *resumed* corpus.

    Both caps can be pinned explicitly and are always honored. When the
    user leaves ``max_attention_tokens`` unset, drop it: at bs=16 a
    single ~1700-frame clip needs 4.6M tokens, and padding-free packing
    raises work per GPU batch. ``max_batch_frames`` bounds the padding
    cost for the rare long clip.
    """
    pinned = int(preprocessing_cfg.get("auto_max_batch_size", 0))
    if pinned > 0:
        max_batch_size = pinned
    if len(files) == 0 or max_batch_size <= 0:
        return max_batch_size, max_batch_frames, max_attention_tokens

    if max_attention_tokens <= 0:
        # No attention cap: packs more per batch and removes the padding tax
        # on the long clips that dominate the remainder of the corpus.
        if max_batch_frames <= 0:
            max_batch_frames = 24000
        return max_batch_size, max_batch_frames, 0
    if max_batch_frames <= 0:
        # Cap padding without capping attention: keeps single-clip batches
        # from ballooning while letting short clips pack together.
        return max_batch_size, 24000, max_attention_tokens
    return max_batch_size, max_batch_frames, max_attention_tokens


def _estimate_mel_frames(path: Path, sample_rate: int, hop_size: int) -> int:
    info = sf.info(str(path))
    frames = int(info.frames)
    if info.samplerate != sample_rate:
        frames = int(np.ceil(frames * sample_rate / info.samplerate))
    return max(1, int(np.ceil(frames / hop_size)))


def _hubert_frames(n_samples: int) -> int:
    return hubert_frame_count(n_samples)


def _load_audio(path: Path, sample_rate: int, resample: str = "soxr_hq") -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type=resample)
    return audio.astype(np.float32)


def _compute_mel_batch(
    audios: list[np.ndarray],
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
    stft_layer: STFT,
    mel_basis: paddle.Tensor,
) -> list[np.ndarray]:
    """GPU Mel：STFT 幅值 -> mel 矩阵乘 -> log10，与旧 CPU 算法仅存在
    可接受的数值误差（卷积实现 STFT、log10 前 clip）。"""
    lengths = [len(audio) for audio in audios]
    max_len = max(lengths)
    padded = np.zeros([len(audios), max_len], dtype="float32")
    for index, audio in enumerate(audios):
        padded[index, : len(audio)] = audio
    source = paddle.to_tensor(padded)
    mag = stft_layer(source, output_format="Magnitude")
    mel = paddle.matmul(mel_basis, mag)
    mel = paddle.log10(paddle.clip(mel, min=1e-5))
    mel_np = mel.numpy()
    frames = [mel_np[index, :, : int(np.ceil(length / hop_size)) + 1] for index, length in enumerate(lengths)]
    return frames


def _load_audio_worker(item: tuple[Path, int, str]) -> np.ndarray:
    """Module-level worker so both thread and process pools can call it."""
    return _load_audio(item[0], item[1], item[2])


def _load_audios(
    batch_files: list[Path],
    sample_rate: int,
    executor: Executor | None = None,
    resample: str = "soxr_hq",
) -> list[np.ndarray]:
    """Load and resample audio files (CPU-bound, parallelizable)."""
    if executor is not None:
        items = [(path, sample_rate, resample) for path in batch_files]
        return list(executor.map(_load_audio_worker, items))
    return [_load_audio(path, sample_rate, resample) for path in batch_files]


def _submit_load(
    batch_files: list[Path],
    sample_rate: int,
    executor: Executor | None,
    resample: str = "soxr_hq",
) -> list | None:
    """Start audio loads in the background so the GPU never waits on them.

    Returns a list of per-file futures (one per file), or None when no pool
    is configured (caller falls back to a synchronous load).
    """
    if executor is None:
        return None
    items = [(path, sample_rate, resample) for path in batch_files]
    return [executor.submit(_load_audio_worker, item) for item in items]


def _prepare_batch(
    audios: list[np.ndarray],
    batch_files: list[Path],
    audio_cfg: dict,
    stft_layer: STFT,
    mel_basis: paddle.Tensor,
) -> list[dict]:
    """GPU mel + sample dict assembly from pre-loaded audio."""
    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    n_mels = int(audio_cfg.get("num_mels", 80))
    n_fft = int(audio_cfg.get("n_fft", 1024))
    hop_size = int(audio_cfg.get("hop_size", 320))
    win_size = int(audio_cfg.get("win_size", 1024))

    mels = _compute_mel_batch(
        audios, sample_rate, n_mels, n_fft, hop_size, win_size, stft_layer, mel_basis
    )
    samples = []
    for path, audio, mel in zip(batch_files, audios, mels):
        n_frames = _hubert_frames(len(audio))
        if n_frames <= 0:
            raise ValueError(f"{path} is too short for HuBERT: {len(audio)} samples")
        samples.append({"path": path, "audio": audio, "mel": mel, "n_frames": n_frames})
    return samples


def _load_batch(
    batch_files: list[Path],
    audio_cfg: dict,
    stft_layer: STFT,
    mel_basis: paddle.Tensor,
    executor: Executor | None = None,
):
    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    resample = str(audio_cfg.get("resample", "soxr_hq"))
    audios = _load_audios(batch_files, sample_rate, executor, resample)
    return _prepare_batch(audios, batch_files, audio_cfg, stft_layer, mel_basis)


def _write_group(
    h5f: h5py.File,
    idx: int,
    sample: dict,
    emb: np.ndarray,
    n_frames: int,
    compression: str | None,
) -> int:
    """Write a single sample to HDF5 and return the new index."""
    grp = h5f.create_group(f"{idx:08d}")
    grp.create_dataset("mel", data=sample["mel"][:, :n_frames], compression=compression)
    grp.create_dataset("hubert", data=emb[:n_frames].astype(np.float32), compression=compression)
    try:
        source_path = str(sample["path"].relative_to(PROJECT_ROOT))
    except ValueError:
        source_path = str(sample["path"])
    grp.attrs["source_path"] = source_path
    grp.attrs["mel_frames"] = n_frames
    grp.attrs["hubert_frames"] = n_frames
    grp.attrs["audio_samples"] = len(sample["audio"])
    return idx + 1


def _work_units(path: Path, sample_rate: int, hop_size: int) -> int:
    frames = _estimate_mel_frames(path, sample_rate, hop_size)
    return max(frames, 1) ** 2


def _completed_groups(h5f: h5py.File) -> int:
    count = 0
    while True:
        key = f"{count:08d}"
        if key not in h5f:
            break
        group = h5f[key]
        if "mel" not in group or "hubert" not in group:
            del h5f[key]
            break
        count += 1
    for key in list(h5f.keys()):
        if key.isdigit() and int(key) >= count:
            del h5f[key]
    return count


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _memory_status() -> str:
    rss_mb = 0
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rss_mb = int(parts[1]) / 1024
                    except ValueError:
                        pass
                break
    anon_mb = _memory_anon_bytes()
    anon_mb = None if anon_mb is None else anon_mb / 1024**2
    limit_bytes = _memory_limit_bytes()
    if anon_mb is not None and limit_bytes is not None and limit_bytes < 2**60:
        return f"rss={rss_mb:.0f}MB anon={anon_mb:.0f}MB limit={limit_bytes / 1024**2:.0f}MB"
    return f"rss={rss_mb:.0f}MB"


def _cgroup_memory_paths() -> tuple[Path, Path] | None:
    candidates = [
        (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
        (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
    ]
    for current, limit in candidates:
        if current.exists() and limit.exists():
            return current, limit
    return None


def _memory_limit_bytes() -> int | None:
    paths = _cgroup_memory_paths()
    if paths is None:
        return None
    value = _read_int(paths[1])
    return None if value is None or value >= 2**60 else value


def _memory_ratio() -> float | None:
    """Return the non-reclaimable (anonymous) memory ratio against the cgroup limit.

    cgroup v1/v2 ``usage`` counters include page cache, which is evicted under
    pressure and never triggers the OOM killer. Using total usage here would
    stop a healthy job long before any real risk, because preprocessing both
    reads hundreds of GB of WAVs and writes many GB of HDF5.
    """
    anon_bytes = _memory_anon_bytes()
    limit_bytes = _memory_limit_bytes()
    if anon_bytes is None or limit_bytes is None:
        return None
    return anon_bytes / limit_bytes


def _memory_anon_bytes() -> int | None:
    """Non-reclaimable resident memory in bytes, ignoring page cache."""
    paths = _cgroup_memory_paths()
    if paths is None:
        return None
    stat_path = paths[0].parent / "memory.stat"
    if not stat_path.exists():
        return None
    try:
        lines = stat_path.read_text().splitlines()
    except OSError:
        return None
    # cgroup v2 reports anonymous memory as "anon"; cgroup v1 reports "rss"
    # (rss is already page-cache-free on v1).
    for key in ("anon", "rss"):
        for line in lines:
            parts = line.split()
            if len(parts) == 2 and parts[0] == key:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def _memory_budget_ratio(config: dict) -> float:
    value = config.get("memory_limit", "auto")
    if isinstance(value, str) and value.lower() == "auto":
        ratio = 0.65 if _memory_limit_bytes() is not None else 0.80
    else:
        ratio = float(value)
    if not 0 < ratio < 1:
        raise ValueError("memory_limit must be 'auto' or a ratio between 0 and 1")
    return ratio


def _ensure_memory_headroom(max_ratio: float) -> None:
    ratio = _memory_ratio()
    if ratio is not None and ratio >= max_ratio:
        raise RuntimeError(
            f"Stopping before cgroup OOM: memory usage is {ratio:.1%}, "
            f"limit is configured at {max_ratio:.1%}. Re-run to resume completed HDF5 groups."
        )


def _run_hubert_batch(model: HubertTeacher, samples: list[dict]) -> list[np.ndarray]:
    with paddle.no_grad():
        feature_list = [
            model.feature_extractor(
                paddle.to_tensor(sample["audio"].reshape([1, 1, -1]))
            ).transpose([0, 2, 1])[0]
            for sample in samples
        ]
        max_frames = max(feature.shape[0] for feature in feature_list)
        features = paddle.zeros([len(samples), max_frames, feature_list[0].shape[1]])
        padding_mask = paddle.ones([len(samples), max_frames], dtype="bool")
        for index, feature in enumerate(feature_list):
            features[index, :feature.shape[0], :] = feature
            padding_mask[index, :feature.shape[0]] = False
        outputs = model.encode_features(features, padding_mask=padding_mask).numpy()
    result = [outputs[i, :feature.shape[0], :].copy() for i, feature in enumerate(feature_list)]
    del feature_list, features, padding_mask, outputs
    return result


def _is_memory_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(token in message for token in ("out of memory", "out_of_memory", "oom", "hip error"))


def _clear_paddle_cache() -> None:
    if paddle.get_device().startswith("gpu"):
        paddle.device.cuda.empty_cache()


def _run_hubert_batch_resilient(model: HubertTeacher, samples: list[dict]) -> list[np.ndarray]:
    try:
        return _run_hubert_batch(model, samples)
    except Exception as error:
        if not _is_memory_error(error) or len(samples) == 1:
            raise
        midpoint = len(samples) // 2
        del error
        gc.collect()
        _clear_paddle_cache()
        return _run_hubert_batch_resilient(model, samples[:midpoint]) + _run_hubert_batch_resilient(
            model, samples[midpoint:]
        )


class HubertPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.audio_cfg = config.get("audio", {})
        self.data_cfg = config.get("data", {})
        self.preprocessing_cfg = config.get("preprocessing", {})

    def _make_loader_pool(self) -> Executor | None:
        """Build the audio-loading pool: threads by default, processes when configured.

        Threads overlap I/O with GPU compute; processes add real CPU parallelism
        for librosa resampling but are only worthwhile when the machine has spare
        cores (this box has 4).
        """
        mode = str(self.preprocessing_cfg.get("loader_mode", "thread"))
        workers = int(self.preprocessing_cfg.get("loader_workers", 0))
        if workers <= 0:
            return None
        if mode == "process":
            return ProcessPoolExecutor(max_workers=workers)
        return ThreadPoolExecutor(max_workers=workers)

    def run(self):
        wavs_dir = _resolve_path(self.data_cfg.get("wavs_dir", "data/libritts/wavs"))
        output_dir = _resolve_path(self.data_cfg.get("hubert_emb_dir", "data/libritts/hubert_embeddings"))
        checkpoint_path = _resolve_path(
            self.data_cfg.get(
                "hubert_checkpoint",
                "ckpts/hubert_teacher/hubert4_paddle_aligned_20260818.pdparams",
            )
        )
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

        sample_rate = int(self.audio_cfg.get("sample_rate", 16000))
        n_mels = int(self.audio_cfg.get("num_mels", 80))
        n_fft = int(self.audio_cfg.get("n_fft", 1024))
        hop_size = int(self.audio_cfg.get("hop_size", 320))
        win_size = int(self.audio_cfg.get("win_size", 1024))

        # GPU Mel/STFT：将频谱计算保持在加速器上，避免每帧回 CPU。
        stft_layer = STFT(
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            output_format="Magnitude",
            verbose=False,
        )
        stft_layer.eval()
        mel_basis = paddle.to_tensor(
            librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels),
            dtype="float32",
        )

        model = HubertTeacher()
        model.load_pretrained(checkpoint_path)
        model.eval()

        max_batch_size = int(self.preprocessing_cfg.get("max_batch_size", 8))
        max_batch_frames = int(self.preprocessing_cfg.get("max_batch_frames", 0))
        max_attention_tokens = int(self.preprocessing_cfg.get("max_attention_tokens", 0))

        self.memory_limit = self.preprocessing_cfg.get("memory_limit", "auto")
        self.memory_limit_ratio = _memory_budget_ratio(self.preprocessing_cfg)

        t0 = time.time()
        ok_train = self._write_split(train_h5, train_files, model, "preprocess-train", stft_layer, mel_basis)
        ok_valid = self._write_split(valid_h5, valid_files, model, "preprocess-valid", stft_layer, mel_basis)

        meta = {
            "total": ok_train + ok_valid,
            "train": ok_train,
            "valid": ok_valid,
            "wavs_dir": str(wavs_dir.relative_to(PROJECT_ROOT) if wavs_dir.is_relative_to(PROJECT_ROOT) else wavs_dir),
            "hubert_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT) if checkpoint_path.is_relative_to(PROJECT_ROOT) else checkpoint_path),
            "paddle_device": paddle.get_device(),
            "max_batch_size": max_batch_size,
            "max_batch_frames": max_batch_frames,
            "max_attention_tokens": max_attention_tokens,
            "memory_limit": self.memory_limit,
            "memory_limit_ratio": self.memory_limit_ratio,
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
        stft_layer: STFT,
        mel_basis: paddle.Tensor,
    ) -> int:
        batch_size, max_batch_frames, max_attention_tokens = _tune_batch_limits(
            files,
            int(self.preprocessing_cfg.get("max_batch_size", 8)),
            int(self.preprocessing_cfg.get("max_batch_frames", 0)),
            int(self.preprocessing_cfg.get("max_attention_tokens", 0)),
            int(self.audio_cfg.get("sample_rate", 16000)),
            int(self.audio_cfg.get("hop_size", 320)),
            self.preprocessing_cfg,
        )
        sample_rate = int(self.audio_cfg.get("sample_rate", 16000))
        hop_size = int(self.audio_cfg.get("hop_size", 320))
        compression = self.preprocessing_cfg.get("hdf5_compression", "gzip")
        compression = None if str(compression).lower() in ("none", "off", "0") else str(compression)
        batches = _batches(
            files,
            batch_size,
            max_batch_frames,
            max_attention_tokens,
            sample_rate,
            hop_size,
        )
        print(
            f"{split_name}: {len(batches)} batches, "
            f"batch_size={batch_size} max_batch_frames={max_batch_frames} "
            f"max_attention_tokens={max_attention_tokens}",
            flush=True,
        )
        ordered_files = [path for batch in batches for path in batch]
        work_units = [_work_units(path, sample_rate, hop_size) for path in ordered_files]
        total_work = sum(work_units)

        ok = 0
        mode = "a" if h5_path.exists() else "w"
        with h5py.File(h5_path, mode) as h5f:
            for key, value in {
                "sample_rate": int(self.audio_cfg.get("sample_rate", 16000)),
                "n_mels": int(self.audio_cfg.get("num_mels", 80)),
                "hop_size": int(self.audio_cfg.get("hop_size", 320)),
                "n_fft": int(self.audio_cfg.get("n_fft", 1024)),
                "win_size": int(self.audio_cfg.get("win_size", 1024)),
            }.items():
                h5f.attrs[key] = value

            ok = _completed_groups(h5f)
            if ok > len(ordered_files):
                raise RuntimeError(f"{h5_path} contains {ok} samples but input has only {len(ordered_files)} files")
            remaining_files = ordered_files[ok:]
            batches = _batches(
                remaining_files,
                batch_size,
                max_batch_frames,
                max_attention_tokens,
                sample_rate,
                hop_size,
            )
            loader_pool = self._make_loader_pool()
            resample = str(self.audio_cfg.get("resample", "soxr_hq"))
            try:
                pbar = tqdm(total=total_work, initial=sum(work_units[:ok]), desc=split_name, unit="work", dynamic_ncols=True)
                start_time = time.time()
                prefetched = None
                for batch_index, batch_files in enumerate(batches):
                    if prefetched is not None:
                        # Submitted one full GPU cycle ago; normally already done.
                        audios = [f.result() for f in prefetched]
                        prefetched = None
                    else:
                        audios = _load_audios(batch_files, sample_rate, loader_pool, resample)
                    # Eagerly start the next batch's audio load while the GPU
                    # works on this batch's mel + HuBERT compute.
                    if batch_index + 1 < len(batches):
                        prefetched = _submit_load(
                            batches[batch_index + 1], sample_rate, loader_pool, resample
                        )
                    _ensure_memory_headroom(self.memory_limit_ratio)
                    samples = _prepare_batch(audios, batch_files, self.audio_cfg, stft_layer, mel_basis)
                    _ensure_memory_headroom(self.memory_limit_ratio)
                    emb = _run_hubert_batch_resilient(model, samples)
                    for i, sample in enumerate(samples):
                        n_frames = min(sample["n_frames"] + 1, emb[i].shape[0], sample["mel"].shape[-1])
                        ok = _write_group(h5f, ok, sample, emb[i], n_frames, compression)
                    h5f.flush()
                    pbar.update(sum(work_units[ok - len(batch_files) : ok]))
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        completed_work = sum(work_units[:ok])
                        pbar.set_postfix(speed=f"{completed_work / elapsed:.1f} work/s", memory=_memory_status())
                    del audios, samples, emb
                    gc.collect()
                pbar.close()
            finally:
                if loader_pool is not None:
                    loader_pool.shutdown(wait=True)

        return ok
