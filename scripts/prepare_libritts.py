"""Prepare LibriTTS data for Conan training.

Usage:
    python scripts/prepare_libritts.py \\
        --src /path/to/datasets/libritts \\
        --out /path/to/柯南唱/datas/libritts

Steps:
    1. Extract all 7 tar.gz files from ``src``
    2. Flatten speaker/chapter structure into ``out/wavs/`` (flat, all .wav files)
    3. (Optional) Run extract_hubert_embs.py to pre-compute embeddings

LibriTTS structure:
    {src}/
    ├── dev-clean.tar.gz
    ├── dev-other.tar.gz
    ├── test-clean.tar.gz
    ├── test-other.tar.gz
    ├── train-clean-100.tar.gz
    ├── train-clean-360.tar.gz
    └── train-other-500.tar.gz

Output:
    {out}/
    ├── wavs/         # All .wav files (flattened, ~586h)
    └── hubert_embeddings/  # (after extract_hubert_embs.py)

For zero-shot training, ConanDataset uses random reference sampling
from the same pool. No separate reference directory needed.
"""

import argparse
import os
import shutil
import tarfile
from pathlib import Path


def extract_tgz(tgz_path: str, dest_dir: str) -> str:
    """Extract a tar.gz file to dest_dir.

    Returns the extracted top-level directory name.
    """
    print(f"  Extracting {Path(tgz_path).name} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    # Return the top-level dir name
    members = os.listdir(dest_dir)
    return dest_dir


def flatten_wavs(src_root: str, out_wavs: str) -> int:
    """Recursively find all .wav files and flatten into out_wavs/.

    LibriTTS has structure: {subset}/{speaker}/{chapter}/*.wav
    This flattens them all into a single flat directory.

    Returns:
        Number of .wav files copied.
    """
    os.makedirs(out_wavs, exist_ok=True)
    count = 0
    for root, dirs, files in os.walk(src_root):
        for f in files:
            if f.endswith(".wav"):
                src = os.path.join(root, f)
                dst = os.path.join(out_wavs, f)
                # Handle filename collisions (unlikely but possible across subsets)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(f)
                    rel_path = os.path.relpath(src, src_root).replace("/", "_").replace("\\", "_")
                    dst = os.path.join(out_wavs, f"{base}_{rel_path}{ext}")
                shutil.copy2(src, dst)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare LibriTTS for Conan")
    parser.add_argument("--src", required=True, help="Directory containing .tar.gz files")
    parser.add_argument("--out", required=True, help="Output directory for Conan")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction (already extracted)")
    parser.add_argument("--keep-extracted", action="store_true",
                        help="Keep extracted intermediate dirs (default: remove)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    extract_dir = out_dir / "raw"
    wavs_dir = out_dir / "wavs"

    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Extract ──
    tgz_files = sorted(src_dir.glob("*.tar.gz"))
    if not tgz_files:
        print("No .tar.gz files found. Use --skip-extract if already extracted.")
        sys.exit(1)

    if not args.skip_extract:
        print(f"Extracting {len(tgz_files)} tar.gz files to {extract_dir} ...")
        for tgz in tgz_files:
            extract_tgz(str(tgz), str(extract_dir))
        print("Extraction complete.")
    else:
        print("Skipping extraction.")

    # ── Step 2: Flatten wavs ──
    search_root = str(extract_dir) if not args.skip_extract else str(src_dir)
    print(f"Flattening .wav files from {search_root} to {wavs_dir} ...")
    count = flatten_wavs(search_root, str(wavs_dir))
    print(f"Flattened {count} .wav files into {wavs_dir}")

    # ── Step 3: Cleanup ──
    if not args.keep_extracted and not args.skip_extract:
        print(f"Removing intermediate extracted dirs: {extract_dir}")
        shutil.rmtree(extract_dir, ignore_errors=True)

    # ── Summary ──
    total_size = sum(f.stat().st_size for f in wavs_dir.glob("*.wav"))
    print(f"\n=== LibriTTS ready at {out_dir} ===")
    print(f"  {count} files, {total_size / 1024**3:.1f} GB")
    print(f"\nNext step: extract HuBERT embeddings")
    print(f"  python scripts/extract_hubert_embs.py \\")
    print(f"      --src {wavs_dir} --out {out_dir}/hubert_embeddings")
    print(f"      --onnx ../hubert4.onnx")


if __name__ == "__main__":
    import sys
    main()
