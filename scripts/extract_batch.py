#!/usr/bin/env python3
"""HuBERT embedding extraction — batched ONNX inference.

Reads audio files sequentially in batches, pads to batch's max length,
runs hubert_batch.onnx, writes embeddings to HDF5.
"""
import os, sys, json, time, argparse
import numpy as np
import soundfile as sf
import h5py
from tqdm import tqdm


def _list_audio(src):
    """Return sorted list of audio file paths."""
    files = sorted(
        os.path.join(src, e.name)
        for e in os.scandir(src)
        if e.name.endswith((".wav", ".flac"))
    )
    return files


def _frames(n):
    """HuBERT frame count from audio sample count."""
    if n < 400:
        return 0
    return (n - 640) // 320 + 1


def _batches(files, batch_size):
    """Yield chunks of files."""
    for i in range(0, len(files), batch_size):
        yield files[i:i + batch_size]


def _load_batch(batch_files):
    """Load audio files in batch, return (audios, lengths, frame_counts)."""
    audios = []
    for f in batch_files:
        try:
            a, _ = sf.read(f, dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            audios.append(a.astype(np.float32))
        except Exception as e:
            print(f"\n[SKIP] {os.path.basename(f)}: {e}", flush=True)
            continue
    return audios


def main():
    p = argparse.ArgumentParser(description="HuBERT batch extraction")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--onnx", default="weights/hubert_batch.onnx")
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0, help="ignored, single-process only")
    args = p.parse_args()

    # ── List files ──
    files = _list_audio(args.src)
    if not files:
        print("[ERROR] No audio files", file=sys.stderr)
        sys.exit(1)
    # Sort by file size (proxy for audio length) so similar-length files batch together
    files.sort(key=lambda f: os.path.getsize(f))
    print(f"{len(files)} files", flush=True)

    # ── Output ──
    train_h5 = os.path.join(args.out, "train.h5")
    valid_h5 = os.path.join(args.out, "valid.h5")
    os.makedirs(args.out, exist_ok=True)
    for h5 in (train_h5, valid_h5):
        if os.path.exists(h5):
            print(f"[ERROR] {h5} exists. Delete manually.", file=sys.stderr)
            sys.exit(1)

    # ── Split ──
    n_valid = min(1000, len(files) // 10)
    train_files = files[:-n_valid]
    valid_files = files[-n_valid:]

    # ── ONNX session ──
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    session = ort.InferenceSession(
        args.onnx,
        providers=[("CUDAExecutionProvider", {"device_id": 0})],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA unavailable")

    # ── Process ──
    def _process(flist, h5_path, name):
        if not flist:
            return 0
        with h5py.File(h5_path, "w") as h5f:
            pbar = tqdm(total=len(flist), desc=name, unit="files", dynamic_ncols=True)
            ok = 0
            t0 = time.time()

            for chunk in _batches(flist, args.max_batch_size):
                # Load (may skip corrupted files)
                audios = _load_batch(chunk)
                if not audios:
                    continue
                N = len(audios)

                # Pad to chunk's max length
                max_len = max(len(a) for a in audios)
                source = np.zeros([N, 1, max_len], dtype=np.float32)
                for i, a in enumerate(audios):
                    source[i, 0, :len(a)] = a

                # Frame count for each file in chunk
                n_frames = np.array([_frames(len(a)) for a in audios], dtype=np.int32)
                T_feat = n_frames.max()
                T_attn = T_feat + 1

                # Mask
                mask = np.ones([N, T_attn], dtype=np.float32)
                for i in range(N):
                    mask[i, :n_frames[i] + 1] = 0.0

                # Inference
                emb = session.run(None, {"source": source, "key_padding_mask": mask})[0]

                # Write
                for i in range(N):
                    grp = h5f.create_group(f"{ok:08d}")
                    grp.create_dataset("hubert", data=emb[i, :n_frames[i] + 1].astype(np.float32),
                                       compression="gzip")
                    ok += 1

                pbar.update(N)
                elapsed = time.time() - t0
                if ok % max(1, len(flist) // 20) < N or ok == len(flist):
                    pbar.set_postfix(speed=f"{ok/elapsed:.1f}/s")

            pbar.close()
            elapsed = time.time() - t0
            print(f"  {name}: {ok} files in {elapsed/60:.1f}min ({ok/elapsed:.1f}/s)", flush=True)
        return ok

    ok_train = _process(train_files, train_h5, "train")
    ok_valid = _process(valid_files, valid_h5, "valid")

    # ── Meta ──
    meta = {
        "total": ok_train + ok_valid,
        "train": ok_train,
        "valid": ok_valid,
        "device": "cuda",
        "batch_model": args.onnx,
    }
    with open(os.path.join(args.out, "conan_meta.json"), "w") as fm:
        json.dump(meta, fm, indent=2)

    print(f"\nDone: {meta['total']} files", flush=True)


if __name__ == "__main__":
    main()
