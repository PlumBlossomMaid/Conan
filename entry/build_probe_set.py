"""Build a small speaker-balanced probe set from the LibriTTS wav directory.

``entry/probe_speaker_leakage.py`` needs several utterances *per speaker*, which
``valid.h5`` cannot provide (150 utterances over 145 speakers — the probe would
score at chance for every feature set, which proves nothing). This picks the
speakers with the most material, takes a spread of their utterances, and links
them into a fresh directory that ``entry/preprocess_hubert.py`` can consume, so
the probe runs with the same mel front-end and HuBERT teacher as training while
giving every speaker enough utterances to appear on both sides of a k-fold split.

The balance is the point; held-out-ness is not, and cannot be. LibriTTS was split
by file, not by speaker, so a speaker rich enough to supply 8 utterances has all
of them in the train split — an earlier version of this docstring claimed the
probe ran on "audio the model never saw", which is false (measured: 0 of 160 held
out). Read the leakage figure as an upper bound, and get held-out accuracy from
``entry/eval_content_extractor.py``.

Symlinks keep this free; the chosen basenames are unique across LibriTTS.

Usage:
    python entry/build_probe_set.py \\
        --wavs-dir /path/to/LibriTTS_audio \\
        --out-dir /path/to/probe_wavs \\
        --speakers 20 --per-speaker 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.libritts_index import list_audio_paths, speaker_of  # noqa: E402


def select_utterances(names: Sequence[str], per_speaker: int) -> List[str]:
    """Pick ``per_speaker`` names spread evenly across a speaker's sorted list.

    Spreading rather than taking the first ``k`` matters because a speaker's
    utterances are usually consecutive sentences of one chapter; an even spread
    samples across the speaker's material instead.
    """
    names = sorted(names)
    if len(names) <= per_speaker:
        return list(names)
    step = len(names) / per_speaker
    return [names[int(index * step)] for index in range(per_speaker)]


def choose_speakers(by_speaker: Dict[str, List[str]], speakers: int, per_speaker: int
                    ) -> List[str]:
    """Speakers with at least ``per_speaker`` utterances, richest first, then by id."""
    eligible = [name for name, items in by_speaker.items() if len(items) >= per_speaker]
    eligible.sort(key=lambda name: (-len(by_speaker[name]), name))
    return eligible[:speakers]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a speaker-balanced probe set.")
    parser.add_argument("--wavs-dir", required=True, help="Source LibriTTS wav directory")
    parser.add_argument("--out-dir", required=True, help="Where to link the chosen wavs")
    parser.add_argument("--speakers", type=int, default=20)
    parser.add_argument("--per-speaker", type=int, default=8)
    parser.add_argument("--manifest-out", default=None,
                        help="Optional JSON listing the chosen files (default: "
                             "<out-dir>/../probe_manifest.json)")
    args = parser.parse_args()

    paths = list_audio_paths(args.wavs_dir)
    by_speaker: Dict[str, List[str]] = {}
    for path in paths:
        by_speaker.setdefault(speaker_of(path.name), []).append(path.name)
    print(f"  {len(paths)} files over {len(by_speaker)} speakers")

    chosen_speakers = choose_speakers(by_speaker, args.speakers, args.per_speaker)
    if len(chosen_speakers) < args.speakers:
        raise SystemExit(
            f"only {len(chosen_speakers)} speakers have >= {args.per_speaker} utterances"
        )

    chosen: List[str] = []
    for speaker in chosen_speakers:
        chosen.extend(select_utterances(by_speaker[speaker], args.per_speaker))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.wavs_dir).resolve()
    for name in chosen:
        link = out_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source_dir / name)

    manifest_path = Path(args.manifest_out) if args.manifest_out else out_dir.parent / "probe_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"filenames": chosen, "speakers": [speaker_of(n) for n in chosen]}, indent=2),
        encoding="utf-8",
    )

    print(f"  speakers={len(chosen_speakers)} x {args.per_speaker} = {len(chosen)} utterances")
    print(f"  linked into {out_dir}")
    print(f"  manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
