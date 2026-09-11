"""Build a discrete content-label codebook from pre-computed HuBERT features.

The Conan paper (arXiv 2507.14534v4) trains the Stream Content Extractor
with a cross-entropy loss against discrete content labels: "Softmax over
J classes per frame, selecting the highest-probability label". Our Stage-1
HDF5 holds continuous 256-dim HuBERT embeddings, so this script clusters a
sample of those frames with k-means and writes the centroid table to disk.
The SCE then distils per-frame labels with the codebook's argmax lookup.

Usage:
    python entry/build_label_codebook.py \
        --num-labels 256 \
        --train-h5 /path/to/train.h5 \
        --valid-h5 /path/to/valid.h5 \
        --out /path/to/codebook.npy

Outputs:
    codebook.npy  (num_labels, 256) float32 centroid table
    codebook.json (num_labels, frame_count, perplexity, hyperparams)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.label_codebook import (
    build_codebook,
    load_frame_sample,
    perplexity,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build HuBERT label codebook (k-means)")
    parser.add_argument("--num-labels", type=int, default=256,
                        help="number of codebook entries J (paper default 256)")
    parser.add_argument("--train-h5", required=True,
                        help="training HDF5 with per-utterance hubert datasets")
    parser.add_argument("--valid-h5", default=None,
                        help="validation HDF5 (optional, adds frames to the sample)")
    parser.add_argument("--out", required=True,
                        help="output .npy path for the centroid table")
    parser.add_argument("--frames", type=int, default=200_000,
                        help="total frame budget drawn for clustering")
    parser.add_argument("--samples-per-file", type=int, default=4,
                        help="frames drawn uniformly per utterance")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Sampling frames from {args.train_h5}"
          + (f" and {args.valid_h5}" if args.valid_h5 else ""), flush=True)
    paths = [args.train_h5] + ([args.valid_h5] if args.valid_h5 else [])
    t0 = time.time()
    features = load_frame_sample(
        paths,
        samples_per_file=args.samples_per_file,
        frames_per_file=args.frames,
    )
    print(f"  sampled {features.shape[0]} frames x {features.shape[1]} dims "
          f"in {time.time() - t0:.1f}s", flush=True)

    print(f"Clustering into J={args.num_labels} labels "
          f"(max_iter={args.max_iter}, n_init={args.n_init}, seed={args.seed})", flush=True)
    t0 = time.time()
    codebook = build_codebook(
        features,
        num_clusters=args.num_labels,
        max_iter=args.max_iter,
        n_init=args.n_init,
        seed=args.seed,
        verbose=True,
    )
    print(f"  k-means done in {time.time() - t0:.1f}s", flush=True)

    labels = codebook.labels(features)
    ppl = perplexity(labels, codebook.num_labels)
    print(f"  codebook perplexity on sample: {ppl:.1f} / {codebook.num_labels} "
          f"({100.0 * ppl / codebook.num_labels:.1f}% active)", flush=True)

    out_path = Path(args.out)
    codebook.save(str(out_path))
    meta = {
        "num_labels": codebook.num_labels,
        "feature_dim": codebook.feature_dim,
        "sample_frames": int(features.shape[0]),
        "perplexity": ppl,
        "max_iter": args.max_iter,
        "n_init": args.n_init,
        "seed": args.seed,
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved codebook -> {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    print(f"Saved meta    -> {json_path}")
    print(f"\nUsage: pass -o data.label_codebook={out_path} to train.py")


if __name__ == "__main__":
    main()
