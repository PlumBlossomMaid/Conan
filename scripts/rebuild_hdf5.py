#!/usr/bin/env python3
"""Rebuild HDF5 with filename keys + HuBERT + mel.

Reads old train.h5/valid.h5 (indexed by file-size order),
copies HuBERT embeddings, reads audio and computes mel via
GPU-batched ppAudio, writes new HDF5 with filename keys.

Usage:
    python scripts/rebuild_hdf5.py

Output:
    datas/libritts/hubert_embeddings/train.h5  (new, old renamed to train_old.h5)
"""

import os, sys, time
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
import paddle
import ppAudio
from tqdm import tqdm

SRC = Path(r"F:\work\konan\datas\libritts\wavs")
OLD_H5_DIR = Path(r"F:\work\konan\datas\libritts\hubert_embeddings")
OUT_H5 = OLD_H5_DIR / "train.h5"

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_SIZE = 320
WIN_SIZE = 1024
BATCH_SIZE = 128  # GPU batch size for mel computation

# ── GPU MelSpectrogram (ppAudio) ──
_mel_extractor = ppAudio.features.MelSpectrogram(
    sr=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_SIZE,
    n_mels=N_MELS, hop_length=HOP_SIZE,
    center=True, power=2.0, fmin=0.0, fmax=None, norm=1,
    trainable_mel=False, trainable_STFT=False, verbose=False,
)


@paddle.no_grad()
def compute_mels_batch(audios: list[np.ndarray]) -> list[np.ndarray]:
    """Batch GPU mel computation.

    Pads all audios in batch to max length, runs GPU MelSpectrogram,
    returns list of (80, T) mels (unpadded).
    """
    max_len = max(len(a) for a in audios)
    B = len(audios)
    batch = np.zeros([B, 1, max_len], dtype=np.float32)
    for i, a in enumerate(audios):
        batch[i, 0, :len(a)] = a
    tensor = paddle.to_tensor(batch)
    mels = _mel_extractor(tensor).numpy()  # (B, n_mels, T_feat)
    # Unpad to each file's actual frame count
    results = []
    for i, a in enumerate(audios):
        n_frames = (len(a) - WIN_SIZE) // HOP_SIZE + 1  # center=True
        if n_frames <= 0:
            n_frames = 1
        results.append(np.log10(np.clip(mels[i, :, :n_frames], 1e-5, None)).astype(np.float32))
    return results


def main():
    # ── 1. List files (by size for old HDF5 mapping) ──
    all_files = sorted(SRC.glob("*.flac")) + sorted(SRC.glob("*.wav"))
    n_total = len(all_files)
    # Sort by size to match old HDF5 key ordering
    files_by_size = sorted(all_files, key=lambda f: os.path.getsize(f))
    name_to_rank = {f.stem: i for i, f in enumerate(files_by_size)}
    print(f"Files: {n_total}", flush=True)

    # ── 2. Backup old train.h5 if exists ──
    old_train = OLD_H5_DIR / "train.h5"
    if old_train.exists():
        backup = OLD_H5_DIR / "train_old.h5"
        if not backup.exists():
            old_train.rename(backup)
            print(f"  Backed up old train.h5 → train_old.h5", flush=True)
        else:
            print(f"  train_old.h5 already exists, keeping both", flush=True)

    # ── 3. Open old HDF5 ──
    old_train_h5_path = OLD_H5_DIR / "train_old.h5"
    if not old_train_h5_path.exists():
        old_train_h5_path = OLD_H5_DIR / "train.h5"  # fallback if no backup
    train_h5 = h5py.File(str(old_train_h5_path), "r")
    valid_h5 = h5py.File(str(OLD_H5_DIR / "valid.h5"), "r")
    n_train = len(train_h5)
    n_valid = len(valid_h5)

    # ── 3. Build output in alphabetical order ──
    alpha_files = sorted(all_files, key=lambda f: f.stem)
    out = h5py.File(str(OUT_H5), "w")
    out.attrs["sample_rate"] = SAMPLE_RATE
    out.attrs["n_mels"] = N_MELS
    out.attrs["hop_size"] = HOP_SIZE
    out.attrs["n_fft"] = N_FFT
    out.attrs["win_size"] = WIN_SIZE

    t0 = time.time()
    ok = 0
    pbar = tqdm(total=n_total, desc="rebuild", unit="files", dynamic_ncols=True)

    # ── 4. Process in GPU batches ──
    for batch_start in range(0, len(alpha_files), BATCH_SIZE):
        batch_files = alpha_files[batch_start:batch_start + BATCH_SIZE]
        audios = []
        batch_names = []
        batch_embs = []

        for fpath in batch_files:
            name = fpath.stem
            rank = name_to_rank[name]
            try:
                audio, _ = sf.read(str(fpath), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(1)
                audios.append(audio)

                # HuBERT from old HDF5
                if rank < n_train:
                    emb = train_h5[f"{rank:08d}"]["hubert"][()]
                else:
                    emb = valid_h5[f"{rank - n_train:08d}"]["hubert"][()]
                batch_embs.append(emb)
                batch_names.append(name)
            except Exception as e:
                print(f"\n[SKIP] {name}: {e}", flush=True)
                pbar.update(1)
                continue

        if not audios:
            continue

        # GPU batched mel
        mels = compute_mels_batch(audios)

        # Write to HDF5
        for i, name in enumerate(batch_names):
            grp = out.create_group(name)
            grp.create_dataset("mel", data=mels[i], compression="gzip")
            grp.create_dataset("hubert", data=batch_embs[i], compression="gzip")
            ok += 1

        pbar.update(len(batch_files))

    out.close()
    train_h5.close()
    valid_h5.close()
    pbar.close()

    elapsed = time.time() - t0
    print(f"\nDone: {ok}/{n_total} files in {elapsed/60:.1f}min", flush=True)
    if ok < n_total:
        print(f"  {n_total - ok} files skipped", flush=True)


if __name__ == "__main__":
    main()
