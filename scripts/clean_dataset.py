#!/usr/bin/env python3
"""Clean corrupted audio files from LibriTTS dataset.

Single-process sf.info() scan for header-level corruption,
only sf.read() confirm on suspicious files. No multiprocessing.
"""
import os, sys, soundfile as sf
from pathlib import Path
from tqdm import tqdm

MAX_SECONDS = 60


def main():
    src = Path(r"F:\work\konan\datas\libritts\wavs")
    files = sorted(src.glob("*.flac")) + sorted(src.glob("*.wav"))
    print(f"Scanning {len(files)} files (single-process)...", flush=True)

    corrupted = []
    for fpath in tqdm(files, desc="scan", unit="files"):
        try:
            info = sf.info(str(fpath))
            if info.frames / info.samplerate > MAX_SECONDS:
                # Suspicious header — confirm with read
                audio, sr = sf.read(str(fpath), dtype="float32")
                if len(audio) / sr > MAX_SECONDS:
                    corrupted.append((fpath, f"too long: {len(audio)/sr:.1f}s"))
        except ValueError as e:
            corrupted.append((fpath, str(e)[:80]))
        except Exception as e:
            corrupted.append((fpath, str(e)[:80]))

    if corrupted:
        print(f"\nBad files ({len(corrupted)}):", flush=True)
        for fpath, err in corrupted:
            fpath.unlink()
            print(f"  DELETED: {fpath.name} — {err}", flush=True)
    else:
        print("No corrupted files found.", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
