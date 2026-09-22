"""Does the extracted "content" still carry the speaker?

Stage 1 is the piece of Conan meant to make timbre swappable: if the content
representation still encodes *who* is talking, swapping a timbre vector cannot
work. ``valid.h5`` cannot answer this — its 150 utterances cover 145 distinct
speakers, so a speaker-classification probe has no training utterance for any
speaker it is tested on and lands at chance for *every* feature set: vacuous.

This runs on a purpose-built set instead (see ``entry/build_probe_set.py``):
S speakers x K utterances, extracted with the same mel front-end and HuBERT
teacher as training. Every speaker contributes to both sides of a stratified
k-fold split, so the probe can learn speakers and the question becomes
meaningful:

    probe accuracy >> chance  ->  the representation leaks speaker identity
    probe accuracy ~= chance  ->  speaker identity is, to this probe, gone

**These utterances are mostly *training* material.** LibriTTS was split by file,
not by speaker, so a speaker rich enough to supply 8 probe utterances almost
certainly has all of his utterances in the train split — measured on this set:
0 of 160 held out. That does not invalidate the probe, but it changes how to
read it:

- the leakage figure is an **upper bound**, since any memorisation of a training
  utterance can only inflate it;
- the comparison against ``target_labels`` (recomputed from audio, never trained
  on) and ``raw_hubert_mean`` is unaffected: same utterances, same protocol;
- the frame accuracy below is therefore *not* a held-out number. Use
  ``entry/eval_content_extractor.py`` for that.

``--held-out-map`` (e.g. ``logs/content_extractor_ce/valid_speakers.json``) makes
the held-out fraction part of the report instead of an assumption.

Feature sets compared, with two self-checks that make the result trustworthy:

===========================  ============================================
random control               must land at chance, else the protocol is broken
predicted content labels     the question: does the model's output leak?
target content labels        the same, for the HuBERT quantiser itself
raw HuBERT mean              what the extractor distils from
mel statistics               positive control: must beat chance, else the
                             probe cannot detect leakage at all
===========================  ============================================

Usage:
    python entry/probe_speaker_leakage.py -c configs/content_extractor_ce.yaml \\
        --ckpt ckpts/content_extractor_ce/last.pdparams \\
        --wavs-dir /path/to/probe_wavs --h5 /path/to/probe/train.h5 \\
        --held-out-map logs/content_extractor_ce/valid_speakers.json \\
        --device gpu --json-out logs/content_extractor_ce/probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.eval_content_extractor import (  # noqa: E402
    bootstrap_ci,
    build_model,
    load_extractor_weights,
    to_numpy,
    topk_accuracy,
)
from layers.label_codebook import PaddleLabelCodebook  # noqa: E402
from utils.config_utils import apply_overrides, load_config  # noqa: E402
from utils.libritts_index import build_mapping, speaker_of  # noqa: E402


# ── Pure helpers (unit-tested in tests/test_probe_speaker_leakage.py) ───────

def stratified_folds(targets: Sequence, folds: int, seed: int = 0) -> np.ndarray:
    """Assign each item to a fold, balanced *within* every class.

    Balancing per class is what makes a k-fold speaker probe meaningful: each
    speaker keeps utterances on both sides of every split.
    """
    targets = np.asarray(targets)
    assignment = np.zeros(targets.size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for label in np.unique(targets):
        index = np.flatnonzero(targets == label)
        assignment[rng.permutation(index)] = np.arange(index.size) % folds
    return assignment


def _design(features: np.ndarray) -> np.ndarray:
    """Append a bias column."""
    return np.hstack([np.asarray(features, dtype=np.float64), np.ones((len(features), 1))])


def fit_softmax(features: np.ndarray, targets: Sequence, iters: int = 600,
                lr: float = 0.5, l2: float = 1e-3) -> np.ndarray:
    """Fit a softmax classifier by full-batch gradient descent (deterministic)."""
    targets = np.asarray(targets).astype(np.int64)
    design = _design(features)
    classes = int(targets.max()) + 1
    weights = np.zeros((design.shape[1], classes), dtype=np.float64)
    onehot = np.zeros((design.shape[0], classes), dtype=np.float64)
    onehot[np.arange(design.shape[0]), targets] = 1.0
    for _ in range(iters):
        logits = design @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        weights -= lr * (design.T @ (probabilities - onehot) / design.shape[0] + l2 * weights)
    return weights


def softmax_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Argmax of the linear scores under a fitted weight matrix."""
    return (_design(features) @ weights).argmax(axis=1)


def probe_cv_accuracy(features: np.ndarray, targets: Sequence, folds: int = 4,
                      seed: int = 0, iters: int = 600) -> float:
    """Out-of-fold speaker-classification accuracy.

    Standardisation uses the *training* folds only — fitting it on all data would
    leak the test folds' statistics and inflate the score.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets).astype(np.int64)
    assignment = stratified_folds(targets, folds, seed)
    hits = total = 0
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        if not test.any() or not train.any():
            continue
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0) + 1e-8
        weights = fit_softmax((features[train] - mean) / std, targets[train], iters=iters)
        predicted = softmax_predict((features[test] - mean) / std, weights)
        hits += int((predicted == targets[test]).sum())
        total += int(test.sum())
    return hits / total if total else 0.0


def label_histogram(sequences: Sequence[np.ndarray], num_labels: int) -> np.ndarray:
    """Per-utterance normalised histogram over labels."""
    histograms = np.zeros((len(sequences), num_labels), dtype=np.float64)
    for index, sequence in enumerate(sequences):
        if sequence.size:
            histograms[index] = np.bincount(sequence, minlength=num_labels) / sequence.size
    return histograms


def majority_share(targets: Sequence) -> float:
    """Accuracy of always predicting the most frequent class."""
    targets = np.asarray(targets)
    if targets.size == 0:
        return 0.0
    counts = np.bincount(targets.astype(np.int64))
    return float(counts.max() / counts.sum())


def held_out_overlap(filenames: Sequence[str], held_out: Sequence[str]) -> Dict[str, object]:
    """How much of a probe set is held-out material, by basename.

    A file-level split gives no held-out utterances to a speaker who supplies
    several, so this is normally ~0 and the leakage figure is an upper bound.
    Reporting it keeps that an observation rather than an assumption.
    """
    names = {Path(name).name for name in filenames}
    held = {Path(name).name for name in held_out}
    overlap = len(names & held)
    return {
        "probe_utterances": len(names),
        "held_out": int(len(held)),
        "probe_utterances_held_out": overlap,
        "held_out_fraction": float(overlap / len(names)) if names else 0.0,
    }


# ── Data and model ─────────────────────────────────────────────────────────

def load_utterances(h5_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Read (mel, hubert) per group in group order, trimmed to the shared length."""
    mels, hubers = [], []
    with h5py.File(h5_path, "r") as handle:
        for key in sorted(handle.keys()):
            mel = np.asarray(handle[key]["mel"], dtype=np.float32)
            hubert = np.asarray(handle[key]["hubert"], dtype=np.float32)
            length = min(mel.shape[-1], hubert.shape[0])
            mels.append(mel[:, :length])
            hubers.append(hubert[:length])
    return mels, hubers


def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker-leakage probe for Stage 1.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--h5", required=True, help="Probe-set HDF5 (mel + hubert)")
    parser.add_argument("--wavs-dir", required=True,
                        help="Directory the probe set was extracted from (for speaker ids)")
    parser.add_argument("--held-out-map", default=None,
                        help="JSON with a 'filenames' list of held-out utterances "
                             "(e.g. logs/.../valid_speakers.json); reports how much of "
                             "the probe set is held out instead of assuming")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("-o", "--override", action="append", default=None, metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.override)
    if args.device:
        paddle.set_device(args.device)
    print(f"  device={paddle.get_device()}")

    sce_cfg = config.get("content_extractor", {})
    num_labels = int(sce_cfg.get("num_labels", 0))
    codebook = PaddleLabelCodebook.load(config["data"]["label_codebook"])
    model = build_model(config, codebook)
    load_extractor_weights(model, args.ckpt)
    model.eval()

    mels, hubers = load_utterances(args.h5)
    print(f"  probe set: {len(mels)} utterances, "
          f"{sum(m.shape[-1] for m in mels)} frames total")

    # Speaker identity: the group's own ``source_path`` attribute is authoritative;
    # ``wavs_dir`` only matters for the replay fallback on older HDF5 files.
    filenames, mapping_report = build_mapping(
        args.h5, wavs_dir=args.wavs_dir, n_valid=len(mels), seed=0
    )
    speakers = [speaker_of(name) for name in filenames]
    speaker_ids = {speaker: index for index, speaker in enumerate(sorted(set(speakers)))}
    targets = np.asarray([speaker_ids[s] for s in speakers])
    print(f"  speakers={len(speaker_ids)} distinct (mapping via {mapping_report['method']})")

    report: Dict[str, object] = {
        "checkpoint": args.ckpt,
        "probe_h5": args.h5,
        "speakers": len(speaker_ids),
        "utterances": len(mels),
        "mapping_validation": mapping_report,
    }
    if args.held_out_map:
        held_out = json.loads(Path(args.held_out_map).read_text(encoding="utf-8"))["filenames"]
        overlap = held_out_overlap(filenames, held_out)
        report["held_out_overlap"] = overlap
        print(f"  held-out material: {overlap['probe_utterances_held_out']}/"
              f"{overlap['probe_utterances']} probe utterances "
              f"({overlap['held_out_fraction']:.1%}) — the leakage figure below is an "
              "upper bound if this is 0")

    # ── Accuracy on the probe set (train material for a file-level split) ──
    print("\n=== Accuracy on the probe set ===")
    predicted_sequences, target_sequences, per_utterance = [], [], []
    all_scores, all_targets = [], []
    for index, (mel, hubert) in enumerate(zip(mels, hubers)):
        length = mel.shape[-1]
        input_tensor = paddle.to_tensor(mel.T[None, :, :])
        logits = to_numpy(model(input_tensor))[0][:length]
        labels = codebook.labels(hubert)
        prediction = logits.argmax(axis=-1)
        predicted_sequences.append(prediction.astype(np.int64))
        target_sequences.append(np.asarray(labels).astype(np.int64))
        per_utterance.append(float((prediction == labels).mean()))
        all_scores.append(logits)
        all_targets.append(np.asarray(labels).astype(np.int64))
    per_utterance = np.asarray(per_utterance)
    low, high = bootstrap_ci(per_utterance)
    flat_scores = np.concatenate(all_scores)
    flat_targets = np.concatenate(all_targets)
    probe_accuracy = {
        "top1_utterance": float(per_utterance.mean()),
        "top1_utterance_ci95": [low, high],
        "top1_frame": float((flat_scores.argmax(axis=1) == flat_targets).mean()),
        "top5_frame": topk_accuracy(flat_scores, flat_targets, 5),
        "frames": int(flat_targets.size),
    }
    report["probe_set_accuracy"] = probe_accuracy
    print(f"  top-1 utterance = {probe_accuracy['top1_utterance']:.4f} "
          f"[95% CI {low:.4f}–{high:.4f}]  (NOT held out — see the module docstring; "
          "the held-out number lives in eval_report_150.json)")
    print(f"  top-1 frame = {probe_accuracy['top1_frame']:.4f}   "
          f"top-5 frame = {probe_accuracy['top5_frame']:.4f}")

    # ── Speaker-leakage probe ──
    print("\n=== Speaker-leakage probe (stratified "
          f"{args.folds}-fold, chance={1.0 / len(speaker_ids):.3f}) ===")
    rng = np.random.default_rng(0)
    feature_sets = {
        "random_control": rng.normal(size=(len(mels), num_labels)),
        "predicted_labels": label_histogram(predicted_sequences, num_labels),
        "target_labels": label_histogram(target_sequences, num_labels),
        "raw_hubert_mean": np.asarray([h.mean(axis=0) for h in hubers]),
        "mel_stats": np.asarray([
            np.concatenate([m.mean(axis=1), m.std(axis=1)]) for m in mels
        ]),
    }
    chance = max(1.0 / len(speaker_ids), majority_share(targets))
    probe = {}
    for name, features in feature_sets.items():
        probe[name] = probe_cv_accuracy(features, targets, folds=args.folds)
        print(f"  {name:18s} probe acc = {probe[name]:.3f}   "
              f"lift over chance = {probe[name] - chance:+.3f}")
    report["speaker_probe"] = {"chance": chance, "folds": args.folds, **probe}

    controls_ok = (
        probe["random_control"] <= chance + 0.1 and probe["mel_stats"] > chance + 0.2
    )
    report["probe_controls_passed"] = bool(controls_ok)
    print(f"  controls: random control at chance={probe['random_control'] <= chance + 0.1}, "
          f"mel positive control beats chance={probe['mel_stats'] > chance + 0.2} "
          f"-> {'VALID' if controls_ok else 'INCONCLUSIVE'}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  report written to {out_path}")


if __name__ == "__main__":
    main()
