#!/usr/bin/env python3
"""Conan HuBERT → HDF5 — multiprocess CUDA via multiprocessing.Pool.

Workers are initialized once with ONNX CUDA session (reused).
Main process collects results via imap_unordered and writes to HDF5.
No shards, no merge step.

Usage:
    python scripts/binarize_conan.py \\
        --src  datas/libritts/wavs \\
        --out  datas/libritts \\
        --onnx ../hubert4.onnx \\
        --workers 4
"""
import argparse, os, sys, json, time, math, pickle
import numpy as np
import soundfile as sf
import h5py
import multiprocessing
from tqdm import tqdm

# Global per-worker: ONNX session
_SESSION = None


def _init_worker(onnx_path: str):
    """Called once per worker — loads ONNX model into CUDA."""
    import onnxruntime as ort
    global _SESSION
    _SESSION = ort.InferenceSession(
        onnx_path,
        providers=[("CUDAExecutionProvider", {"device_id": 0})],
    )
    if "CUDAExecutionProvider" not in _SESSION.get_providers():
        raise RuntimeError("CUDAExecutionProvider unavailable — check CUDA/cuDNN installation")


def _worker(wav_path: str):
    """Process one file: load audio → HuBERT inference → return embedding."""
    global _SESSION
    try:
        audio, _ = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        if len(audio) < 3200:  # < 0.2s
            return None
        feats = audio.reshape(1, 1, -1).astype(np.float32)
        emb = _SESSION.run(None, {"source": feats})[0][0].astype(np.float32)
        return emb
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--onnx", default="../hubert4.onnx")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    onnx_path = args.onnx if os.path.exists(args.onnx) else \
        os.path.join(os.path.dirname(__file__), "..", args.onnx)

    # List all audio files sorted
    files = sorted(
        os.path.join(args.src, e.name)
        for e in os.scandir(args.src)
        if e.name.endswith((".wav", ".flac"))
    )
    print(f"{len(files)} files ({args.workers} workers, CUDA)")
    sys.stdout.flush()

    if not files:
        return

    # Check for existing HDF5 files (resume)
    train_h5 = os.path.join(args.out, "train.h5")
    valid_h5 = os.path.join(args.out, "valid.h5")
    skip_count = 0
    if os.path.exists(train_h5) and os.path.exists(valid_h5):
        with h5py.File(train_h5, "r") as f:
            skip_count += len(f)
        with h5py.File(valid_h5, "r") as f:
            skip_count += len(f)
        print(f"  Already complete: {skip_count} items. Delete .h5 to re-run.")
        return

    # Remove any partial output from previous runs
    for h5 in (train_h5, valid_h5):
        if os.path.exists(h5):
            os.remove(h5)

    # Verify CUDA availability BEFORE spawning workers
    import subprocess
    _r = subprocess.run(
        [sys.executable, "-c",
         f"import onnxruntime as ort; ort.get_available_providers()"
         f".index('CUDAExecutionProvider')"],
        capture_output=True, text=True, timeout=30,
    )
    if _r.returncode != 0:
        _err = _r.stderr.strip().split("\n")[-1] if _r.stderr else "CUDA unavailable"
        print(f"  ❌ {_err}", flush=True)
        print("  Set PATH to include nvidia/cublas/bin;nvidia/cuda_runtime/bin;nvidia/cudnn/bin")
        sys.exit(1)

    # ── Multiprocess extraction ──
    t0 = time.time()
    ok = err = 0
    valid_embeddings = []  # last 1000 items go here

    # Reserve last 1000 items for validation
    n_valid = min(1000, len(files) // 10)
    train_files = files[:-n_valid] if n_valid else files
    valid_files = files[-n_valid:] if n_valid else []

    # Open HDF5 (append mode — writes as results come in)
    with h5py.File(train_h5, "w") as train_f, \
         h5py.File(valid_h5, "w") as valid_f:

        # Process training set
        if train_files:
            train_idx = 0
            t0 = time.time()
            pool = multiprocessing.Pool(
                args.workers,
                initializer=_init_worker,
                initargs=(onnx_path,),
            )
            try:
                pbar = tqdm(pool.imap_unordered(_worker, train_files, chunksize=10),
                            total=len(train_files), desc="train", unit="files")
                for emb in pbar:
                    if emb is not None:
                        grp = train_f.create_group(f"{train_idx:08d}")
                        grp.create_dataset("hubert", data=emb, compression="gzip")
                        train_idx += 1
                        ok += 1
                    else:
                        err += 1
                    pbar.set_postfix(ok=ok, err=err)
                pbar.close()
            except Exception as e:
                print(f"\n  ❌ Pool broken: {e}. Workers NOT respawned.", flush=True)
                raise
            finally:
                pool.terminate()
                pool.join()
                pool.close()

        # Process validation set
        if valid_files:
            valid_idx = 0
            pool = multiprocessing.Pool(
                args.workers,
                initializer=_init_worker,
                initargs=(onnx_path,),
            )
            try:
                pbar = tqdm(pool.imap_unordered(_worker, valid_files, chunksize=10),
                            total=len(valid_files), desc="valid", unit="files")
                for emb in pbar:
                    if emb is not None:
                        grp = valid_f.create_group(f"{valid_idx:08d}")
                        grp.create_dataset("hubert", data=emb, compression="gzip")
                        valid_idx += 1
                        ok += 1
                    else:
                        err += 1
                    pbar.set_postfix(ok=ok, err=err)
                pbar.close()
            except Exception as e:
                print(f"\n  ❌ Pool broken: {e}. Workers NOT respawned.", flush=True)
                raise
            finally:
                pool.terminate()
                pool.join()
                pool.close()

    elapsed = time.time() - t0
    print(f"\nDone: {ok}+{err} in {elapsed/60:.1f}min ({ok/elapsed:.1f}/s)")
    sys.stdout.flush()

    # Metadata
    with h5py.File(train_h5, "r") as f:
        n_train = len(f)
    with h5py.File(valid_h5, "r") as f:
        n_valid = len(f)
    meta = {"total": ok, "train": n_train, "valid": n_valid}
    with open(os.path.join(args.out, "conan_meta.json"), "w") as fm:
        json.dump(meta, fm, indent=2)
    print(f"  {json.dumps(meta)}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
