"""Resolve which LibriTTS utterance — and speaker — each HDF5 group came from.

``entry/preprocess_hubert.py`` writes ``mel`` and ``hubert`` per group, and
records the originating wav in the group's ``source_path`` attribute. That
attribute is authoritative: reading it gives an exact, ambiguity-free mapping
from group index to file (and hence to speaker, the leading field of a LibriTTS
filename).

For HDF5 files written before that attribute existed, the split can still be
replayed because it is fully determined by the preprocessing code:

    files = every ``*.wav`` under the wavs dir, sorted by file size
    valid = random.Random(valid_seed).sample(files, n_valid), then sorted by size
    group i of valid.h5  ==  valid[i]

A replayed order can be *checked* against the HDF5: groups were written in size
order, so the stored frame counts must be non-decreasing, and each frame count
must predict its file's size (LibriTTS is 24 kHz 16-bit mono, i.e. 960 bytes per
20 ms frame). ``build_mapping`` reports which method it used and, for a replay,
returns that validation so callers know how far to trust it. Byte-size ties are
order-ambiguous and are counted rather than hidden.

Usage (writes the mapping plus the report as JSON):

    python -m utils.libritts_index \\
        --h5 /path/to/valid.h5 --out logs/content_extractor_ce/valid_speakers.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_SUFFIXES = {".wav"}
# LibriTTS is 24 kHz 16-bit mono: 24000 samples/s * 2 bytes / 50 frames/s.
BYTES_PER_FRAME = 960.0
WAV_HEADER_BYTES = 44


def speaker_of(filename: str) -> str:
    """LibriTTS speaker id — the leading underscore-separated field."""
    return str(filename).split("_")[0]


def source_paths(h5_path) -> List[str]:
    """Recorded source path of every group, in group-name order.

    Raises:
        KeyError: If any group lacks the ``source_path`` attribute, which means
            the file predates it and the split has to be replayed instead.
    """
    with h5py.File(h5_path, "r") as handle:
        paths = []
        for key in sorted(handle.keys()):
            recorded = handle[key].attrs.get("source_path")
            if recorded is None:
                raise KeyError(f"group {key} of {h5_path} has no 'source_path' attribute")
            paths.append(str(recorded))
        return paths


def group_frames(h5_path) -> List[int]:
    """Mel frame count of every group, in group-name order."""
    with h5py.File(h5_path, "r") as handle:
        return [int(handle[key]["mel"].shape[-1]) for key in sorted(handle.keys())]


def list_audio_paths(wavs_dir) -> List[Path]:
    """Mirror ``entry/preprocess_hubert._list_audio``: audio files sorted by size."""
    files = [
        path for path in Path(wavs_dir).iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    files.sort(key=lambda path: path.stat().st_size)
    return files


def reconstruct_valid_files(paths: Sequence[Path], n_valid: int, seed: int) -> List[Path]:
    """Replay the split: sample ``n_valid`` paths, then sort them by file size."""
    sampled = random.Random(seed).sample(list(paths), n_valid)
    return sorted(sampled, key=lambda path: path.stat().st_size)


def validate_mapping(frames_per_group: Sequence[int], sizes: Sequence[int]) -> Dict[str, object]:
    """Check a replayed ordering against the HDF5 frame counts.

    Two independent signals: the groups were written in file-size order, so the
    frame counts must be non-decreasing, and each frame count predicts its file's
    byte size. Ties in byte size are order-ambiguous, so they are counted and
    reported rather than hidden.
    """
    frames = np.asarray(frames_per_group, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    residual = np.abs(sizes - (WAV_HEADER_BYTES + frames * BYTES_PER_FRAME))
    _, counts = np.unique(sizes, return_counts=True)
    return {
        "groups": int(frames.size),
        "frames_non_decreasing": bool((np.diff(frames) >= 0).all()) if frames.size > 1 else True,
        "size_frame_correlation": float(np.corrcoef(sizes, frames)[0, 1]) if frames.size > 1 else 1.0,
        "residual_median_bytes": float(np.median(residual)),
        "residual_max_bytes": int(residual.max()),
        "residual_median_frames": float(np.median(residual) / BYTES_PER_FRAME),
        "ambiguous_groups": int(counts[counts > 1].sum()) if counts.size else 0,
    }


def build_mapping(h5_path, wavs_dir=None, n_valid: Optional[int] = None, seed: int = 1234
                  ) -> Tuple[List[str], Dict[str, object]]:
    """Resolve every HDF5 group to its source filename.

    Args:
        h5_path: HDF5 file whose groups are in the preprocessor's write order.
        wavs_dir: Only needed to replay the split when ``source_path`` is absent.
        n_valid: Only needed for the replay (the ``n_valid`` used at preprocessing).
        seed: Only needed for the replay (the ``valid_seed`` used at preprocessing).

    Returns:
        ``(filenames, report)`` where ``filenames[i]`` is the group ``i`` source
        file and ``report["method"]`` is ``"source_path"`` (authoritative) or
        ``"replay"`` (validated reconstruction).
    """
    try:
        names = [Path(path).name for path in source_paths(h5_path)]
    except KeyError:
        if wavs_dir is None or n_valid is None:
            raise ValueError(
                f"{h5_path} has no source_path attributes; pass wavs_dir and n_valid "
                "to replay the preprocessing split instead"
            ) from None
        frames = group_frames(h5_path)
        if len(frames) != n_valid:
            raise ValueError(
                f"{h5_path} has {len(frames)} groups but n_valid={n_valid}; "
                "pass the n_valid used at preprocessing time"
            )
        candidates = reconstruct_valid_files(list_audio_paths(wavs_dir), n_valid, seed)
        names = [path.name for path in candidates]
        return names, {
            "method": "replay",
            "groups": len(names),
            "distinct_speakers": len({speaker_of(name) for name in names}),
            "wavs_dir": str(wavs_dir),
            "h5_path": str(h5_path),
            "n_valid": int(n_valid),
            "valid_seed": int(seed),
            **validate_mapping(frames, [path.stat().st_size for path in candidates]),
        }

    return names, {
        "method": "source_path",
        "groups": len(names),
        "distinct_speakers": len({speaker_of(name) for name in names}),
        "h5_path": str(h5_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", required=True, help="HDF5 produced by preprocess_hubert")
    parser.add_argument("--wavs-dir", default=None, help="Only for the replay fallback")
    parser.add_argument("--n-valid", type=int, default=None, help="Only for the replay fallback")
    parser.add_argument("--seed", type=int, default=1234, help="Only for the replay fallback")
    parser.add_argument("--out", required=True, help="JSON output path")
    args = parser.parse_args()

    filenames, report = build_mapping(args.h5, args.wavs_dir, args.n_valid, args.seed)
    payload = {
        "filenames": filenames,
        "speakers": [speaker_of(name) for name in filenames],
        "validation": report,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  method={report['method']}  groups={report['groups']}  "
          f"distinct speakers={report['distinct_speakers']}")
    if report["method"] == "replay":
        print(f"  frames non-decreasing : {report['frames_non_decreasing']}")
        print(f"  corr(size, frames)    : {report['size_frame_correlation']:.6f}")
        print(f"  order-ambiguous groups: {report['ambiguous_groups']}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()
