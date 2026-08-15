"""LibriTTS-style dataset for Conan training.

Supports:
    - Raw waveform → mel-spectrogram + F0
    - HuBERT content labels (from pre-computed HDF5 or ONNX placeholder)
    - Chunked streaming data for online training simulation

HuBERT labels will be computed via ONNX Runtime in production.
Currently supports pre-computed HuBERT labels from HDF5.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import librosa
import numpy as np
import onnxruntime as ort
import paddle
import soundfile as sf
from paddle.io import Dataset


class ConanDataset(Dataset):
    """Conan training dataset.

    Each item is a pair of (source_audio, source_hubert_labels, reference_audio).

    Args:
        data_dir: Root data directory with subdirs ``source`` and ``reference``.
        sample_rate: Audio sample rate (default 16000).
        n_mels: Number of mel bins (default 80).
        n_fft: FFT size (default 1024).
        hop_size: STFT hop size (default 320).
        win_size: STFT window size (default 1024).
        f0_min: Minimum F0 for pitch extraction (default 40).
        f0_max: Maximum F0 for pitch extraction (default 1100).
        chunk_size: Training chunk size in frames (default 50 = 1s at 50Hz).
        hubert_onxx_path: Path to HuBERT ONNX model (optional, for future use).
        hubert_emb_dir: Directory with pre-computed HuBERT 256-dim embeddings (.npy files).
        is_train: Whether this is training (random sampling) or validation.
        max_samples: Maximum number of samples (for debugging).
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_size: int = 320,
        win_size: int = 1024,
        f0_min: float = 40.0,
        f0_max: float = 1100.0,
        chunk_size: int = 50,
        hubert_onnx_path: Optional[str] = None,
        hubert_emb_dir: Optional[str] = None,
        hdf5_path: Optional[str] = None,
        compute_f0: bool = False,
        is_train: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.chunk_size = chunk_size
        self.hubert_onnx_path = hubert_onnx_path
        self.hubert_emb_dir = Path(hubert_emb_dir) if hubert_emb_dir else None
        self.hdf5_path = hdf5_path
        self._h5f = None  # lazy HDF5 handle (reopened per worker after pickling)
        self.compute_f0 = compute_f0
        self.is_train = is_train

        # ── Build file index ──
        self.samples: List[Dict] = []

        # Expect structure: data_dir/{source,reference}/*.wav
        source_dir = self.data_dir / "source"
        ref_dir = self.data_dir / "reference"

        # If data_dir/wavs/*.wav exists, use single-dir mode
        wav_dir = self.data_dir / "wavs"
        if wav_dir.exists():
            source_files = sorted(wav_dir.glob("*.wav")) + sorted(wav_dir.glob("*.flac"))
        elif source_dir.exists():
            source_files = sorted(source_dir.glob("*.wav")) + sorted(source_dir.glob("*.flac"))
        else:
            raise FileNotFoundError(
                f"Neither {wav_dir} nor {source_dir} found in {data_dir}"
            )

        # Filter to files with pre-computed HuBERT embeddings (only for .npy mode)
        if self.hubert_emb_dir is not None and self.hdf5_path is None:
            emb_stems = {p.stem for p in self.hubert_emb_dir.glob("*.npy")}
            filtered = [f for f in source_files if f.stem in emb_stems]
            skipped = len(source_files) - len(filtered)
            if skipped:
                print(f"  Filtered {skipped} files without HuBERT embeddings")
            source_files = filtered

        if max_samples:
            source_files = source_files[:max_samples]

        # For zero-shot VC, each training step uses a random reference speaker.
        # We store all file paths; the training script will sample references.
        self.source_files = [str(f) for f in source_files]

        # Reference files (same pool, different speaker)
        if ref_dir.exists():
            self.ref_files = sorted([str(f) for f in ref_dir.glob("*.wav")]) + sorted([str(f) for f in ref_dir.glob("*.flac")])
        else:
            self.ref_files = self.source_files

        if len(self.source_files) == 0:
            raise RuntimeError(f"No audio files found in {data_dir}")

        self.sizes = [self._estimate_frames(path) for path in self.source_files]

        print(f"  ConanDataset: {len(self.source_files)} source files, {len(self.ref_files)} ref files")

        # STFT and mel basis (lazy init in __getitem__)
        self._mel_basis: Optional[np.ndarray] = None
        self._stft_window: Optional[np.ndarray] = None

        # HuBERT ONNX runtime (placeholder for future)
        self._hubert_session = None
        if hubert_onnx_path:
            print(f"  [Info] HuBERT ONNX available at: {hubert_onnx_path} (not used in training)")

    def _init_hubert_onnx(self, model_path: str):
        """Initialize HuBERT ONNX Runtime session (CPU, to coexist with Paddle GPU)."""
        self._hubert_session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider']
        )
        print(f"  HuBERT ONNX loaded (CPU): {model_path}")

    def _estimate_frames(self, path: str) -> int:
        info = sf.info(path)
        frames = int(info.frames)
        if info.samplerate != self.sample_rate:
            frames = int(np.ceil(frames * self.sample_rate / info.samplerate))
        return max(1, int(np.ceil(frames / self.hop_size)))

    def num_frames(self, idx: int) -> int:
        return self.sizes[idx]

    def _get_mel_basis(self) -> np.ndarray:
        if self._mel_basis is None:
            self._mel_basis = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
            ).T
        return self._mel_basis

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-spectrogram from audio.

        Args:
            audio: (T,) waveform.

        Returns:
            mel: (n_mels, T_mel) float32.
        """
        spec = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window="hann",
            center=True,
        )
        mag = np.abs(spec)  # (F, T)
        mel = np.dot(self._get_mel_basis().T, mag)  # (n_mels, T)
        mel = np.log10(np.clip(mel, 1e-5, None))
        return mel.astype(np.float32)

    def _compute_f0(self, audio: np.ndarray, hop_size: int) -> np.ndarray:
        """Compute F0 contour using librosa pyin.

        Args:
            audio: (T,) waveform.
            hop_size: Hop size in samples.

        Returns:
            f0: (T_f0,) float32 F0 values (0 for unvoiced).
        """
        # librosa.pyin crashes on this machine (numba JIT segfault).
        # Using numpy autocorrelation instead.
        frame_length = hop_size * 2
        n_frames = len(audio) // hop_size + 1
        f0 = np.zeros(n_frames, dtype=np.float32)

        min_period = int(self.sample_rate / self.f0_max)
        max_period = int(self.sample_rate / self.f0_min)
        if max_period <= min_period:
            return f0

        for i in range(n_frames):
            start = i * hop_size
            end = start + frame_length
            if end > len(audio):
                end = len(audio)
            frame = audio[start:end]
            if len(frame) < max_period + 1:
                continue

            frame = frame - frame.mean()
            # Autocorrelation
            r = np.correlate(frame, frame, mode='full')[len(frame)-1:]
            energy = r[0]  # zero-lag energy
            r[:min_period] = 0  # zero out below min frequency
            # keep r[0] for threshold but set to 0 after saving
            r[0] = 0
            r[max_period+1:] = 0  # zero out above max frequency

            peak_idx = np.argmax(r)
            if energy > 1e-10 and r[peak_idx] > 0.3 * energy:
                f0[i] = self.sample_rate / peak_idx

        return f0

    def _load_audio(self, path: str) -> np.ndarray:
        """Load and resample audio.

        Args:
            path: Audio file path.

        Returns:
            audio: (T,) float32 in [-1, 1].
        """
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)

    def _load_hubert_embedding(self, idx: int) -> Optional[np.ndarray]:
        """Load pre-computed HuBERT 256-dim embedding from HDF5 or .npy.

        Priority: HDF5 (faster for many files) > .npy (legacy).

        Args:
            idx: Sample index.

        Returns:
            emb: (T_emb, 256) float32 or None if not available.
        """
        # HDF5 mode: open file lazily (each DataLoader worker gets its own handle)
        if self.hdf5_path:
            if self._h5f is None:
                self._h5f = h5py.File(self.hdf5_path, 'r')
            key = f"{idx:08d}"
            if key not in self._h5f:
                return None
            return self._h5f[key]["hubert"][()]

        # Legacy .npy mode
        if self.hubert_emb_dir is None:
            return None
        src_path = Path(self.source_files[idx])
        emb_path = self.hubert_emb_dir / f"{src_path.stem}.npy"
        if emb_path.exists():
            return np.load(str(emb_path)).astype(np.float32)
        return None

    def __len__(self) -> int:
        return len(self.source_files)

    def __getitem__(self, idx: int) -> Dict:
        """Get a training sample.

        Returns a dict with:
            - source_audio: (T_audio,) source waveform.
            - source_mel: (n_mels, T_mel) source mel-spectrogram.
            - ref_audio: (T_audio,) reference waveform (different speaker).
            - ref_mel: (n_mels, T_mel) reference mel-spectrogram.
            - f0: (T_mel,) source F0 contour.
            - hubert_emb: (T_emb, 256) HuBERT continuous embedding (if available).
            - source_path: Source file path (for debugging).
        """
        source_audio = self._load_audio(self.source_files[idx])

        # Sample a different reference (for zero-shot training)
        ref_idx = random.choice([i for i in range(len(self.ref_files)) if i != idx])
        ref_audio = self._load_audio(self.ref_files[ref_idx])

        # Compute mel-spectrograms
        source_mel = self._compute_mel(source_audio)
        ref_mel = self._compute_mel(ref_audio)

        # Compute F0
        f0 = self._compute_f0(source_audio, self.hop_size)

        # Align lengths
        mel_len = source_mel.shape[-1]
        f0_len = f0.shape[0]
        if f0_len > mel_len:
            f0 = f0[:mel_len]
        elif f0_len < mel_len:
            f0 = np.pad(f0, (0, mel_len - f0_len), mode="constant")

        # Load HuBERT embedding (256-dim continuous, no clustering)
        # Falls back to on-the-fly ONNX CPU computation when pre-computed .npy not available
        hubert_emb = self._load_hubert_embedding(idx)

        return {
            "source_audio": source_audio,
            "source_mel": source_mel,
            "ref_audio": ref_audio,
            "ref_mel": ref_mel,
            "f0": f0,
            "hubert_emb": hubert_emb,
            "source_path": self.source_files[idx],
        }

    def collater(self, samples: List[Dict]) -> Dict:
        """Collate batch of samples with padding.

        Args:
            samples: List of dicts from __getitem__.

        Returns:
            Batched dict with:
                - source_audio: (B, T_max_audio) padded source audio.
                - source_mel: (B, n_mels, T_max) padded source mel.
                - ref_mel: (B, n_mels, T_ref_max) padded ref mel.
                - f0: (B, T_max) padded F0.
                - hubert_emb: (B, T_max, 256) padded HuBERT embeddings (or None).
                - lengths: (B,) original lengths in mel frames.
        """
        # Sort by source mel length (descending for efficient packing)
        samples.sort(key=lambda s: s["source_mel"].shape[-1], reverse=True)

        source_mels = [s["source_mel"] for s in samples]
        source_audios = [s["source_audio"] for s in samples]
        ref_mels = [s["ref_mel"] for s in samples]
        f0s = [s["f0"] for s in samples]

        # Pad source mel
        max_mel_len = max(m.shape[-1] for m in source_mels)
        padded_mels = []
        for m in source_mels:
            pad_len = max_mel_len - m.shape[-1]
            if pad_len > 0:
                m = np.pad(m, [(0, 0), (0, pad_len)], mode="constant", constant_values=-11.5)
            padded_mels.append(m)

        # Pad source audio
        max_audio_len = max(a.shape[0] for a in source_audios)
        padded_audios = []
        for a in source_audios:
            pad_len = max_audio_len - a.shape[0]
            if pad_len > 0:
                a = np.pad(a, (0, pad_len), mode="constant")
            padded_audios.append(a)

        # Pad ref mel
        max_ref_len = max(m.shape[-1] for m in ref_mels)
        padded_ref_mels = []
        for m in ref_mels:
            pad_len = max_ref_len - m.shape[-1]
            if pad_len > 0:
                m = np.pad(m, [(0, 0), (0, pad_len)], mode="constant", constant_values=-11.5)
            padded_ref_mels.append(m)

        # Pad F0
        pad_shared_len = max_mel_len
        padded_f0s = []
        for f in f0s:
            pad_len = pad_shared_len - f.shape[0]
            if pad_len > 0:
                f = np.pad(f, (0, pad_len), mode="constant")
            padded_f0s.append(f)

        # HuBERT embeddings (256-dim continuous, padded to max time dim)
        has_hubert = samples[0].get("hubert_emb") is not None
        if has_hubert:
            padded_embs = []
            for s in samples:
                emb = s["hubert_emb"]
                if emb is None:
                    emb = np.zeros([pad_shared_len, 256], dtype=np.float32)
                T_emb = emb.shape[0]
                if T_emb > pad_shared_len:
                    emb = emb[:pad_shared_len, :]
                elif T_emb < pad_shared_len:
                    emb = np.pad(emb, [(0, pad_shared_len - T_emb), (0, 0)], mode="constant")
                padded_embs.append(emb)

        lengths = np.array([m.shape[-1] for m in source_mels], dtype=np.int64)

        batch = {
            "source_audio": paddle.to_tensor(np.stack(padded_audios, axis=0), dtype=paddle.float32),
            "source_mel": paddle.to_tensor(np.stack(padded_mels, axis=0), dtype=paddle.float32),
            "ref_mel": paddle.to_tensor(np.stack(padded_ref_mels, axis=0), dtype=paddle.float32),
            "f0": paddle.to_tensor(np.stack(padded_f0s, axis=0), dtype=paddle.float32).unsqueeze(-1),
            "lengths": paddle.to_tensor(lengths),
        }
        if has_hubert:
            batch["hubert_emb"] = paddle.to_tensor(np.stack(padded_embs, axis=0), dtype=paddle.float32)
        return batch


class ContentExtractorDataset(Dataset):
    """Minimal dataset for Stage 1 Content Extractor training.

    Reads mel + HuBERT embeddings from HDF5 (filename-keyed).
    No audio I/O, no librosa.
    """

    def __init__(
        self,
        hdf5_path: str,
        max_frames: int = 500,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.max_frames = max_frames
        self.hdf5_path = hdf5_path
        self._h5f = None

        # List keys from HDF5. Mel frame counts are read from the dataset shapes
        # (metadata only, no sample data) so ConanBatchSampler can batch by frames.
        with h5py.File(hdf5_path, 'r') as f:
            keys = sorted(f.keys())
            if max_samples:
                keys = keys[:max_samples]
            self.sizes = [f[k]["mel"].shape[-1] for k in keys]
        self.keys = keys
        print(f"  ContentExtractorDataset: {len(self.keys)} files")

    def __len__(self):
        return len(self.keys)

    def num_frames(self, idx: int) -> int:
        """Frames this item costs in a batch.

        ``__getitem__`` crops or pads every item to ``max_frames``, so the cost
        is constant regardless of the source length. A frame budget therefore
        derives the batch size from ``max_frames`` rather than varying it per
        batch; ``sizes`` still reflects true source lengths for sorting.
        """
        return self.max_frames

    def __getitem__(self, idx):
        if self._h5f is None:
            self._h5f = h5py.File(self.hdf5_path, 'r')
        key = self.keys[idx]
        mel = self._h5f[key]["mel"][()]          # (80, T)
        hubert_emb = self._h5f[key]["hubert"][()]  # (T, 256)

        # ── Random window crop ──
        T = mel.shape[-1]
        if T > self.max_frames:
            start = random.randint(0, T - self.max_frames)
            mel = mel[:, start:start + self.max_frames]
            h_start = min(start, len(hubert_emb) - 1) if start < len(hubert_emb) else 0
            h_end = min(h_start + self.max_frames, len(hubert_emb))
            hubert_emb = hubert_emb[h_start:h_end]
        elif T < self.max_frames:
            pad = self.max_frames - T
            mel = np.pad(mel, [(0, 0), (0, pad)], mode="constant", constant_values=-11.5)
            h_pad = self.max_frames - len(hubert_emb)
            if h_pad > 0:
                hubert_emb = np.pad(hubert_emb, [(0, h_pad), (0, 0)], mode="constant")

        return {"mel": mel, "hubert_emb": hubert_emb}

    def collater(self, samples):
        mels = np.stack([s["mel"] for s in samples], axis=0).astype(np.float32)
        M = mels.shape[-1]

        # Always produce hubert_emb, fill missing with zeros
        embs = []
        for s in samples:
            emb = s["hubert_emb"]
            if emb is None:
                emb = np.zeros([M, 256], dtype=np.float32)
            T = emb.shape[0]
            if T < M:
                emb = np.pad(emb, [(0, M - T), (0, 0)], mode="constant")
            elif T > M:
                emb = emb[:M]
            embs.append(emb)

        return {
            "source_mel": mels,
            "hubert_emb": np.stack(embs, axis=0).astype(np.float32),
        }
