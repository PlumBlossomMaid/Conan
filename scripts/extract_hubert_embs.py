#!/usr/bin/env python3
"""Extract HuBERT 256-dim embeddings — multiprocess CUDA.

Parent does NOT import onnxruntime — only os/sys for task scheduling.
Each child process independently imports onnxruntime + soundfile + creates
its own CUDA session. No CUDA fork conflicts.

Usage:
    python scripts/extract_hubert_embs.py \\
        --src datas/libritts/wavs \\
        --out datas/libritts/hubert_embeddings \\
        --onnx ../hubert4.onnx \\
        --workers 3
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _worker(args):
    """Worker: imports onnxruntime+soundfile (child only). CUDA inference."""
    import numpy as np
    import soundfile as sf
    import onnxruntime as ort

    wav_path, out_path, onnx_path = args
    if os.path.exists(out_path):
        return "skip", wav_path

    try:
        audio, _ = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(1)
    except Exception as e:
        return "error", f"{wav_path}: load {e}"
    if len(audio) < 160:
        return "skip", wav_path

    feats = audio.reshape(1, 1, -1).astype(np.float32)
    try:
        session = ort.InferenceSession(
            onnx_path,
            providers=[('CUDAExecutionProvider', {'device_id': 0}),
                       'CPUExecutionProvider']
        )
        emb = session.run(None, {"source": feats})[0]
    except Exception as e:
        return "error", f"{wav_path}: inf {e}"

    np.save(out_path, emb[0].astype(np.float32))
    return "ok", wav_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--onnx", default="../hubert4.onnx")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    onnx_path = args.onnx if os.path.exists(args.onnx) else \
        os.path.join(os.path.dirname(__file__), "..", args.onnx)

    # discover files (scandir)
    wavs = sorted(e.name for e in os.scandir(args.src)
                  if e.name.endswith(('.wav', '.flac', '.WAV', '.FLAC')))
    if not wavs:
        print(f"No audio in {args.src}")
        return

    existing = {e.name.replace('.npy', '') for e in os.scandir(args.out)
                if e.name.endswith('.npy')}

    pending = [(os.path.join(args.src, f), os.path.join(args.out, f.rsplit('.',1)[0] + '.npy'), onnx_path)
               for f in wavs if f.rsplit('.',1)[0] not in existing]
    print(f"{len(wavs)} total, {len(wavs)-len(pending)} done, {len(pending)} pending ({args.workers} workers)")
    sys.stdout.flush()

    if not pending:
        return

    # Process in batches to avoid queue deadlock
    BATCH = 5000
    t0 = time.time()
    ok = skip = err = 0

    for bstart in range(0, len(pending), BATCH):
        batch = pending[bstart:bstart + BATCH]
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            fut_map = {exe.submit(_worker, p): p for p in batch}
            for fut in as_completed(fut_map):
                s, msg = fut.result()
                if s == "ok": ok += 1
                elif s == "skip": skip += 1
                else: err += 1

                done = ok + skip + err
                if done % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    eta = (len(pending) - done) / rate if rate else 0
                    print(f"  [{done}/{len(pending)}] {ok}ok {skip}skip {err}err  "
                          f"{rate:.1f}/s  ETA {eta/3600:.1f}h", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {ok}+{skip}+{err} in {elapsed/60:.1f}min ({len(pending)/elapsed:.1f}/s)", flush=True)


if __name__ == "__main__":
    main()
