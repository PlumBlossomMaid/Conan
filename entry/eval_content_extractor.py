"""Offline evaluation of the Stage-1 Stream Content Extractor (CE mode).

``val/acc`` alone is hard to interpret: content labels are strongly
autocorrelated in time, so a trivial "copy the previous frame" predictor can
score well. This script re-runs the trained checkpoint over the held-out HDF5
and reports, in four parts:

1. Held-out metrics  — top-1 / top-5 / macro accuracy, per-class support and
   accuracy, the confusion matrix, and codebook-usage perplexity.
2. Baseline controls — random guess, the most frequent label, and the
   previous-frame (persistence) predictor, so ``val/acc`` reads in context.
3. Input ablations   — shuffle / zero / noise / reverse the mel frames. A model
   that actually uses its input must collapse on all four. The noise condition
   is the only one that preserves frame/label alignment, so it is additionally
   split by whether the clean pass was already right (``residual_report``):
   surviving content keeps the clean predictions informative, whereas a
   fallback output distribution is independent of them.
4. Causality + speed — perturbing future frames must not change outputs for
   frames older than the configured right-context lookahead; plus a
   full-context RTF measurement on real samples.

The dataset, dataloader and logit/target alignment mirror
``models.content_extractor.ContentExtractorModel`` exactly, so the top-1
number here should reproduce the final ``val/acc`` from training.

Usage:
    python entry/eval_content_extractor.py -c configs/content_extractor_ce.yaml \
        --ckpt ckpts/content_extractor_ce/last.pdparams \
        -o data.val_hdf5_path=/path/to/valid.h5 \
        -o data.label_codebook=data/libritts/codebook_256.npy \
        --device gpu --json-out logs/content_extractor_ce/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.dataset import ContentExtractorDataset  # noqa: E402
from layers.label_codebook import PaddleLabelCodebook, perplexity  # noqa: E402
from layers.stream_content_extractor import StreamContentExtractor  # noqa: E402
from utils.config_utils import apply_overrides, load_config  # noqa: E402
from utils.training_utils import build_val_dataloader  # noqa: E402


# ── Metric helpers (pure numpy; unit-tested in tests/) ──────────────────────

def to_tensor(x, dtype="float32") -> paddle.Tensor:
    """Coerce a dataloader field (numpy or paddle) to a float paddle tensor."""
    if isinstance(x, paddle.Tensor):
        return x
    return paddle.to_tensor(np.asarray(x), dtype=dtype)


def to_numpy(x) -> np.ndarray:
    """Coerce a paddle tensor or array-like to numpy."""
    if isinstance(x, paddle.Tensor):
        return x.numpy()
    return np.asarray(x)


def valid_frames(scores: np.ndarray, target: np.ndarray, mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten ``(B, T, ...)`` and keep only frames where ``mask`` is set.

    Args:
        scores: (B, T, C) per-frame scores.
        target: (B, T) int64 labels.
        mask: (B, T) validity mask.

    Returns:
        (N, C) scores and (N,) targets for the N valid frames.
    """
    scores = np.asarray(scores)
    keep = np.asarray(mask).reshape(-1) > 0
    return scores.reshape(-1, scores.shape[-1])[keep], np.asarray(target).reshape(-1)[keep]


def topk_accuracy(scores: np.ndarray, target: np.ndarray, k: int) -> float:
    """Fraction of rows whose top-``k`` scores contain the target."""
    if scores.shape[0] == 0:
        return 0.0
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return float((idx == target.reshape(-1, 1)).any(axis=1).mean())


def label_sequences(labels: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """Split a padded batch into the leading valid label run of each sample."""
    sequences = []
    for row_labels, row_mask in zip(np.asarray(labels), np.asarray(mask)):
        length = int((np.asarray(row_mask) > 0).sum())
        if length > 0:
            sequences.append(np.asarray(row_labels)[:length].astype(np.int64))
    return sequences


def previous_frame_accuracy(sequences: Sequence[np.ndarray]) -> float:
    """Accuracy of predicting the previous frame's label (persistence baseline)."""
    hits = total = 0
    for sequence in sequences:
        if sequence.size < 2:
            continue
        hits += int((sequence[1:] == sequence[:-1]).sum())
        total += sequence.size - 1
    return hits / total if total else 0.0


def most_frequent_accuracy(sequences: Sequence[np.ndarray], num_labels: int) -> float:
    """Accuracy of always predicting the most frequent label in the set."""
    if not sequences:
        return 0.0
    counts = np.bincount(np.concatenate(sequences), minlength=num_labels)
    return float(counts.max() / counts.sum())


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_labels: int) -> np.ndarray:
    """(num_labels, num_labels) confusion matrix, rows = target, cols = pred."""
    index = np.asarray(target).astype(np.int64) * num_labels + np.asarray(pred).astype(np.int64)
    return np.bincount(index, minlength=num_labels * num_labels).reshape(num_labels, num_labels)


def per_class_stats(conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (support, accuracy) per class from a confusion matrix."""
    support = conf.sum(axis=1)
    accuracy = np.divide(conf.diagonal(), np.maximum(support, 1))
    return support, accuracy


def per_utterance_accuracy(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Top-1 accuracy of each utterance over its own valid frames.

    This is the quantity the trainer logs as ``val/acc``: Ocean averages the
    per-batch accuracy and the validation dataloader uses batch_size=1, so the
    logged number is a mean over utterances — not over frames.
    """
    accuracy = []
    for row_scores, row_labels, row_mask in zip(scores, labels, mask):
        length = int((np.asarray(row_mask) > 0).sum())
        if length == 0:
            continue
        hits = row_scores[:length].argmax(axis=1) == row_labels[:length]
        accuracy.append(float(hits.mean()))
    return np.asarray(accuracy, dtype=np.float64)


def residual_report(clean_pred: np.ndarray, ablated_pred: np.ndarray, labels: np.ndarray,
                    mask: np.ndarray) -> Dict[str, float]:
    """Split an ablation's surviving accuracy by whether the clean pass was right.

    An ablation can score above chance for two very different reasons: some of the
    frame's content survived the perturbation, or the model simply fell back on a
    narrowed output distribution that happens to overlap the target marginal. The
    two are separable without any extra forward pass, because the clean pass is
    already available: surviving *content* keeps the clean predictions informative
    (frames the clean pass got right are far more likely to survive), while a
    fallback distribution is independent of them.

    The independence baseline for ``hits_shared_with_clean`` is the clean accuracy
    itself, and for ``lift`` it is zero.
    """
    keep = np.asarray(mask) > 0
    clean_pred = np.asarray(clean_pred)[keep]
    ablated_pred = np.asarray(ablated_pred)[keep]
    labels = np.asarray(labels)[keep]
    if clean_pred.size == 0:
        return {"frames": 0, "ablated_accuracy": 0.0, "clean_accuracy": 0.0,
                "accuracy_given_clean_correct": 0.0, "accuracy_given_clean_wrong": 0.0,
                "lift": 0.0, "hits_shared_with_clean": 0.0}
    clean_ok = clean_pred == labels
    ablated_ok = ablated_pred == labels
    hits = int(ablated_ok.sum())
    return {
        "frames": int(clean_pred.size),
        "ablated_accuracy": float(ablated_ok.mean()),
        "clean_accuracy": float(clean_ok.mean()),
        "accuracy_given_clean_correct": float(ablated_ok[clean_ok].mean()) if clean_ok.any() else 0.0,
        "accuracy_given_clean_wrong": float(ablated_ok[~clean_ok].mean()) if (~clean_ok).any() else 0.0,
        "lift": (float(ablated_ok[clean_ok].mean()) if clean_ok.any() else 0.0)
                - (float(ablated_ok[~clean_ok].mean()) if (~clean_ok).any() else 0.0),
        "hits_shared_with_clean": float((ablated_ok & clean_ok).sum()) / hits if hits else 0.0,
    }


def bootstrap_ci(values: Sequence[float], reps: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling utterances (clusters).

    Utterances, not frames, are the independent unit: frames inside one utterance
    are highly correlated, so a frame-level CI would be far too narrow.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(reps, values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def duration_bucket(lengths: Sequence[int], edges: Sequence[float]) -> np.ndarray:
    """Bucket index per frame count, given ascending interval edges."""
    return np.digitize(np.asarray(lengths, dtype=np.float64), np.asarray(edges)[1:-1])


# ── Model / data plumbing ──────────────────────────────────────────────────

def build_model(config: dict, codebook: PaddleLabelCodebook) -> StreamContentExtractor:
    """Instantiate the extractor exactly as ``ContentExtractorModel`` does."""
    sce_cfg = config.get("content_extractor", {})
    audio_cfg = config.get("audio", {})
    return StreamContentExtractor(
        input_dim=audio_cfg.get("num_mels", 80),
        d_model=sce_cfg.get("d_model", 512),
        nhead=sce_cfg.get("nhead", 8),
        num_layers=sce_cfg.get("num_layers", 6),
        output_dim=sce_cfg.get("output_dim", 256),
        chunk_size=sce_cfg.get("chunk_size", 4),
        left_context=sce_cfg.get("left_context", 1),
        right_context=sce_cfg.get("right_context", 2),
        dim_feedforward=sce_cfg.get("dim_feedforward", 2048),
        dropout=sce_cfg.get("dropout", 0.1),
        num_labels=int(sce_cfg.get("num_labels", 0)),
        label_codebook=codebook,
    )


def load_extractor_weights(model: paddle.nn.Layer, ckpt_path: str, prefix: str = "extractor.") -> None:
    """Load ``<prefix>*`` weights from an Ocean checkpoint into ``model``.

    Ocean checkpoints wrap everything in ``state_dict``; the Stage-1 model
    keeps its network under the ``extractor`` submodule. Keys are *structured*
    names (``mel_proj.0.weight``), so ``set_state_dict`` needs
    ``use_structured_name=True`` — passing ``False`` expects Paddle's generated
    names (``linear_0.w_0``) and silently loads nothing, which would make the
    evaluation report chance-level accuracy as if it were a finding. A key
    mismatch therefore raises instead.
    """
    checkpoint = paddle.load(ckpt_path)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not stripped:
        raise ValueError(
            f"no parameters with prefix {prefix!r} in {ckpt_path} "
            f"(available: {list(state_dict)[:5]}...)"
        )
    missing, unexpected = model.set_state_dict(stripped, use_structured_name=True)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint {ckpt_path} does not match the model built from the config: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys "
            f"(missing e.g. {list(missing)[:3]}, unexpected e.g. {list(unexpected)[:3]})"
        )
    print(f"  loaded {len(stripped)} tensors from {ckpt_path} (prefix={prefix!r})")


def align_scores(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Crop/pad ``(B, T_pred, C)`` scores to the target length, as training does."""
    t_pred, t_target = scores.shape[1], np.asarray(target).shape[1]
    if t_pred > t_target:
        return scores[:, :t_target]
    if t_pred < t_target:
        return np.pad(scores, [(0, 0), (0, t_target - t_pred), (0, 0)])
    return scores


@paddle.no_grad()
def collect_scores(
    model: paddle.nn.Layer,
    loader,
    transform: Optional[Callable[[paddle.Tensor, dict], paddle.Tensor]] = None,
) -> Dict[str, np.ndarray]:
    """Run the model over ``loader``, optionally perturbing the mel input.

    Args:
        model: The content extractor in eval mode.
        loader: Validation dataloader yielding the CE-mode batch dict.
        transform: Optional ``(mel, batch) -> mel`` applied before the forward
            pass (used for the input ablations).

    Returns:
        Dict with ``scores`` (B, T, C), ``labels`` (B, T), ``mask`` (B, T) and
        ``lengths`` (B,).
    """
    scores_out, labels_out, mask_out, lengths_out = [], [], [], []
    for batch in loader:
        mel = to_tensor(batch["source_mel"])
        if transform is not None:
            mel = transform(mel, batch)
        scores = model(mel.transpose([0, 2, 1]))
        labels = to_numpy(batch["hubert_label"])
        scores_out.append(align_scores(to_numpy(scores), labels))
        labels_out.append(labels)
        mask_out.append(to_numpy(batch["valid_mask"]))
        lengths_out.append(to_numpy(batch["lengths"]))
    return {
        "scores": np.concatenate(scores_out, axis=0),
        "labels": np.concatenate(labels_out, axis=0),
        "mask": np.concatenate(mask_out, axis=0),
        "lengths": np.concatenate(lengths_out, axis=0),
    }


# ── Per-checkpoint scoring ─────────────────────────────────────────────────

def utterance_features(hdf5_path) -> Dict[str, np.ndarray]:
    """Per-utterance probe features read straight from the HDF5, in group order.

    The CE-mode dataloader yields labels only, so the raw HuBERT embedding and the
    mel are read here. The valid region is the frames both arrays share, matching
    ``ContentExtractorDataset.__getitem__``, and the index order matches the
    dataloader's ``sorted(keys)`` order.

    Returns:
        Dict with ``hubert_mean`` (N, 256), ``mel_stats`` (N, 2 * n_mels —
        per-mel-bin mean and standard deviation over time) and ``frames`` (N,).
    """
    hubert_mean, mel_stats, frames = [], [], []
    with h5py.File(hdf5_path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            mel = np.asarray(group["mel"], dtype=np.float32)
            hubert = np.asarray(group["hubert"], dtype=np.float32)
            length = min(mel.shape[-1], hubert.shape[0])
            mel_valid = mel[:, :length]
            hubert_mean.append(hubert[:length].mean(axis=0))
            mel_stats.append(np.concatenate([mel_valid.mean(axis=1), mel_valid.std(axis=1)]))
            frames.append(length)
    return {
        "hubert_mean": np.asarray(hubert_mean),
        "mel_stats": np.asarray(mel_stats),
        "frames": np.asarray(frames),
    }


def core_metrics(model, loader, num_labels: int, bootstrap_seed: int = 0
                 ) -> Tuple[Dict[str, object], Dict[str, np.ndarray], np.ndarray]:
    """Score one checkpoint over the loader.

    Returns:
        The metrics dict (frame top-1, utterance top-1 with a bootstrap CI, frame
        top-5 and codebook-usage perplexities), the raw collected scores, and the
        per-utterance accuracies so callers can stratify them.
    """
    collected = collect_scores(model, loader)
    scores, targets = valid_frames(collected["scores"], collected["labels"], collected["mask"])
    per_utterance = per_utterance_accuracy(
        collected["scores"], collected["labels"], collected["mask"]
    )
    low, high = bootstrap_ci(per_utterance, seed=bootstrap_seed)
    predicted = scores.argmax(axis=1)
    metrics = {
        "utterances": int(per_utterance.size),
        "frames": int(targets.size),
        "top1_frame": float((predicted == targets).mean()),
        "top1_utterance": float(per_utterance.mean()),
        "top1_utterance_ci95": [low, high],
        "top5_frame": topk_accuracy(scores, targets, 5),
        "pred_perplexity": perplexity(predicted, num_labels),
        "target_perplexity": perplexity(targets, num_labels),
    }
    return metrics, collected, per_utterance


# ── Input ablations ────────────────────────────────────────────────────────

def _valid_region(mel: paddle.Tensor, batch: dict) -> Tuple[paddle.Tensor, int]:
    """Return a copy of ``mel`` plus the valid frame count of the first sample."""
    return mel.clone(), int(to_numpy(batch["lengths"])[0])


def make_ablation(kind: str, seed: int = 0) -> Callable[[paddle.Tensor, dict], paddle.Tensor]:
    """Build a mel perturbation used to test whether the model uses its input.

    Args:
        kind: One of ``shuffle``, ``zero``, ``noise`` or ``reverse``.
        seed: RNG seed for the shuffle permutation and the noise draw.

    Returns:
        A ``transform(mel, batch) -> mel`` callable operating on the valid
        (unpadded) region of the first batch sample.
    """
    rng = np.random.default_rng(seed)

    def transform(mel: paddle.Tensor, batch: dict) -> paddle.Tensor:
        out, length = _valid_region(mel, batch)
        if length <= 1:
            return out
        region = out[:, :, :length]
        if kind == "shuffle":
            perm = paddle.to_tensor(rng.permutation(length).astype(np.int64))
            out[:, :, :length] = paddle.index_select(region, perm, axis=2)
        elif kind == "zero":
            out[:, :, :length] = paddle.zeros_like(region)
        elif kind == "noise":
            std = float(region.std().item()) or 1.0
            noise = paddle.to_tensor(rng.normal(0.0, std, size=region.shape).astype(np.float32))
            out[:, :, :length] = region + noise
        elif kind == "reverse":
            out[:, :, :length] = paddle.flip(region, axis=[2])
        else:
            raise ValueError(f"unknown ablation {kind!r}")
        return out

    return transform


# ── Causality and latency ──────────────────────────────────────────────────

@paddle.no_grad()
def causality_check(model: paddle.nn.Layer, loader, chunk_size: int, right_context: int
                    ) -> Dict[str, object]:
    """Verify that only the configured right-context lookahead can be non-causal.

    ``EmformerEncoder.forward`` gives chunk ``i`` the chunks ``[i-1, i+2]`` as
    its attention window (``left_context``/``right_context``), so a frame may
    depend on future frames only up to ``right_context`` chunks. This overwrites
    every mel frame from a chunk-aligned midpoint onwards with a large constant
    and checks that the scores of the provably-unaffected prefix are bit-identical
    while the frames just before the cut do change.

    Args:
        model: The content extractor in eval mode.
        loader: Validation dataloader.
        chunk_size: Frames per Emformer chunk.
        right_context: Chunks of lookahead the architecture is allowed.

    Returns:
        Dict with the cut frame, the number of provably-unaffected prefix frames,
        the max absolute score change there, and the change at the last frame
        before the cut (non-zero whenever a lookahead is configured).
    """
    batch = max((b for b in loader), key=lambda b: int(to_numpy(b["lengths"])[0]))
    mel = to_tensor(batch["source_mel"])
    length = int(to_numpy(batch["lengths"])[0])
    # Align the cut to a chunk boundary so the affected-chunk arithmetic is exact.
    cut = max(chunk_size, (length // 2 // chunk_size) * chunk_size)
    # Chunk C is perturbed, and chunk j sees chunk C when j - 1 <= C <= j + right_context,
    # i.e. every chunk j >= C - right_context is affected.
    affected_from = max(0, (cut // chunk_size - right_context) * chunk_size)

    base = to_numpy(model(mel.transpose([0, 2, 1])))[0]
    perturbed = mel.clone()
    perturbed[:, :, cut:] = 5.0  # mel is log10-scaled, so this is far out of range
    after = to_numpy(model(perturbed.transpose([0, 2, 1])))[0]

    prefix_delta = float(np.abs(base[:affected_from] - after[:affected_from]).max()) if affected_from else 0.0
    boundary_delta = float(np.abs(base[cut - 1:cut] - after[cut - 1:cut]).max())
    return {
        "sample_frames": length,
        "cut_frame": cut,
        "chunk_size": chunk_size,
        "right_context": right_context,
        "lookahead_frames": right_context * chunk_size,
        "prefix_frames_checked": affected_from,
        "prefix_max_delta": prefix_delta,
        "last_frame_before_cut_delta": boundary_delta,
        "prefix_unchanged": prefix_delta <= 1e-5,
    }


@paddle.no_grad()
def measure_rtf(model: paddle.nn.Layer, loader, chunk_size: int,
                warmup: int = 3, repeat: int = 10) -> Dict[str, float]:
    """Measure full-context real-time factor on a real validation sample.

    The sample is trimmed to its own length and padded up to a chunk multiple,
    so the timing excludes the dataset's fixed 500-frame padding.

    Args:
        model: The content extractor in eval mode.
        loader: Validation dataloader.
        chunk_size: Frames per Emformer chunk.
        warmup: Untimed warmup iterations.
        repeat: Timed iterations.

    Returns:
        Dict with latency in ms, the audio duration used, and the RTF.
    """
    batch = max((b for b in loader), key=lambda b: int(to_numpy(b["lengths"])[0]))
    mel = to_tensor(batch["source_mel"])
    length = int(to_numpy(batch["lengths"])[0])
    padded = int(np.ceil(length / chunk_size) * chunk_size)
    x = paddle.zeros([1, padded, mel.shape[1]])
    x[:, :length, :] = mel[:, :, :length].transpose([0, 2, 1])

    for _ in range(warmup):
        model(x)
    paddle.device.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        model(x)
    paddle.device.synchronize()
    latency_ms = (time.perf_counter() - start) / repeat * 1000

    audio_s = length * 0.02  # one content frame per 20ms at 50Hz
    return {
        "frames": float(length),
        "audio_seconds": audio_s,
        "latency_ms": latency_ms,
        "rtf": latency_ms / (audio_s * 1000),
        "ms_per_frame": latency_ms / max(length, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Stage-1 CE content extractor on held-out data."
    )
    parser.add_argument("-c", "--config", required=True, help="Stage config YAML")
    parser.add_argument("--ckpt", required=True, help="Checkpoint (.pdparams)")
    parser.add_argument("--device", default=None,
                        help="paddle device, e.g. cpu / gpu. Omit to use paddle's "
                             "default (portable).")
    parser.add_argument("--json-out", default=None, help="Write the report as JSON here")
    parser.add_argument("--ablation-seed", type=int, default=0)
    parser.add_argument("--val-max-samples", type=int, default=None,
                        help="Evaluate this many held-out utterances instead of "
                             "data.val_max_samples (valid.h5 holds 150).")
    parser.add_argument("--speaker-map", default=None,
                        help="JSON from utils/libritts_index.py — enables the "
                             "per-speaker breakdown.")
    parser.add_argument("--extra-ckpt", action="append", default=None,
                        metavar="PATH", help="Also score this checkpoint, repeatable "
                                             "(gives a checkpoint-accuracy curve).")
    parser.add_argument("-o", "--override", action="append", default=None,
                        metavar="KEY=VALUE", help="Config override, repeatable")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.override)
    if args.val_max_samples is not None:
        config.setdefault("data", {})["val_max_samples"] = args.val_max_samples
    if args.device:
        paddle.set_device(args.device)
    print(f"  device={paddle.get_device()}  config={args.config}")

    data_cfg = config.get("data", {})
    sce_cfg = config.get("content_extractor", {})
    num_labels = int(sce_cfg.get("num_labels", 0))
    if num_labels <= 0:
        raise ValueError("this evaluator is for CE mode; set content_extractor.num_labels > 0")

    codebook = PaddleLabelCodebook.load(data_cfg["label_codebook"])
    dataset = ContentExtractorDataset(
        hdf5_path=data_cfg.get("val_hdf5_path", data_cfg.get("hdf5_path")),
        max_frames=config.get("audio", {}).get("max_frames", 500),
        max_samples=data_cfg.get("val_max_samples", 50),
        label_codebook=codebook,
    )
    loader = build_val_dataloader(dataset)

    model = build_model(config, codebook)
    load_extractor_weights(model, args.ckpt)
    model.eval()

    report: Dict[str, object] = {"checkpoint": args.ckpt, "config": args.config}

    # ── 1. Held-out metrics and baseline controls ──
    print("\n=== 1. Held-out metrics + baseline controls ===")
    metrics, clean, per_utterance = core_metrics(model, loader, num_labels, args.ablation_seed)
    scores, targets = valid_frames(clean["scores"], clean["labels"], clean["mask"])
    sequences = label_sequences(clean["labels"], clean["mask"])
    predicted_sequences = label_sequences(clean["scores"].argmax(axis=-1), clean["mask"])
    pred = scores.argmax(axis=1)
    conf = confusion_matrix(pred, targets, num_labels)
    support, class_acc = per_class_stats(conf)
    frame_counts = [int(sequence.size) for sequence in sequences]

    metrics.update({
        "class_macro_acc": float(class_acc[support > 0].mean()),
        "classes_seen": int((support > 0).sum()),
        "frames_min": int(min(frame_counts)),
        "frames_max": int(max(frame_counts)),
        "baseline_random": 1.0 / num_labels,
        "baseline_most_frequent": most_frequent_accuracy(sequences, num_labels),
        "baseline_previous_frame": previous_frame_accuracy(sequences),
    })
    report["metrics"] = metrics

    pairs = [(int(support[i]), float(class_acc[i])) for i in range(num_labels)]
    worst = sorted(range(num_labels), key=lambda i: -pairs[i][0])[:8]
    order = np.argsort(-conf, axis=None)
    rows, cols = np.unravel_index(order, conf.shape)
    report["top_confusions"] = [
        {"target": int(t), "predicted": int(p), "count": int(conf[t, p])}
        for t, p in zip(rows, cols) if t != p
    ][:8]
    report["per_class_worst"] = [
        {"label": int(i), "support": pairs[i][0], "accuracy": pairs[i][1]}
        for i in worst if pairs[i][0] > 0
    ]
    low, high = metrics["top1_utterance_ci95"]
    print(f"  utterances={metrics['utterances']}  frames={metrics['frames']} "
          f"(len {metrics['frames_min']}–{metrics['frames_max']} frames)")
    print(f"  top-1 utterance = {metrics['top1_utterance']:.4f}  [95% CI {low:.4f}–{high:.4f}]"
          "   <- the quantity the trainer logs as val/acc")
    print(f"  top-1 frame     = {metrics['top1_frame']:.4f}")
    print(f"  top-5 frame     = {metrics['top5_frame']:.4f}")
    print(f"  class macro-acc = {metrics['class_macro_acc']:.4f}   "
          f"classes hit={metrics['classes_seen']}/{num_labels}")
    print(f"  perplexity: pred={metrics['pred_perplexity']:.1f}  "
          f"target={metrics['target_perplexity']:.1f} (max {num_labels})")
    print(f"  baselines : random={metrics['baseline_random']:.4f}  "
          f"most-frequent={metrics['baseline_most_frequent']:.4f}  "
          f"previous-frame={metrics['baseline_previous_frame']:.4f}")
    print("  worst classes (label, support, acc): "
          + ", ".join(f"{d['label']}({d['support']},{d['accuracy']:.2f})"
                      for d in report["per_class_worst"]))
    print("  top confusions (target->pred, count): "
          + ", ".join(f"{d['target']}->{d['predicted']}({d['count']})"
                      for d in report["top_confusions"][:5]))

    # ── 2. Stratification by duration and speaker ──
    print("\n=== 2. Stratification ===")
    edges = (0, 100, 200, 400, 10 ** 9)
    bucket_index = duration_bucket(clean["lengths"], edges)
    report["duration_buckets"] = []
    for index in range(len(edges) - 1):
        selected = per_utterance[bucket_index == index]
        if not selected.size:
            continue
        report["duration_buckets"].append({
            "frames_from": int(edges[index]),
            "frames_to": int(edges[index + 1]),
            "utterances": int(selected.size),
            "top1_utterance": float(selected.mean()),
        })
    print("  by duration: " + ", ".join(
        f"{b['frames_from']}-{b['frames_to']}f: n={b['utterances']}, "
        f"acc={b['top1_utterance']:.3f}" for b in report["duration_buckets"]))

    if args.speaker_map:
        payload = json.loads(Path(args.speaker_map).read_text(encoding="utf-8"))
        speakers = np.asarray(payload["speakers"])
        if speakers.size < per_utterance.size:
            raise ValueError(
                f"speaker map holds {speakers.size} entries but {per_utterance.size} "
                "utterances were evaluated; rebuild it with the same n_valid"
            )
        speakers = speakers[: per_utterance.size]
        per_speaker = {}
        for speaker in sorted(set(speakers.tolist())):
            selected = per_utterance[speakers == speaker]
            per_speaker[speaker] = {"utterances": int(selected.size),
                                    "top1_utterance": float(selected.mean())}
        report["per_speaker"] = per_speaker
        report["speaker_map_validation"] = payload.get("validation")
        repeated = sum(1 for value in per_speaker.values() if value["utterances"] > 1)
        print(f"  by speaker: {len(per_speaker)} speakers over {per_utterance.size} "
              f"utterances ({repeated} speakers repeat) — far too few repeats for a "
              "per-speaker claim; use the dedicated probe for that")

    # ── 3. Input ablations ──
    print("\n=== 3. Input ablations (accuracy must collapse) ===")
    ablations, ablated_predicted, ablated_collected = {}, {}, {}
    for kind in ("shuffle", "zero", "noise", "reverse"):
        collected = collect_scores(model, loader, make_ablation(kind, args.ablation_seed))
        ablated_scores, ablated_targets = valid_frames(
            collected["scores"], collected["labels"], collected["mask"]
        )
        ablations[kind] = float((ablated_scores.argmax(axis=1) == ablated_targets).mean())
        ablated_predicted[kind] = label_sequences(
            collected["scores"].argmax(axis=-1), collected["mask"]
        )
        ablated_collected[kind] = collected
        print(f"  {kind:8s} top-1 = {ablations[kind]:.4f}")
    report["ablations"] = ablations

    # The noise condition scores well above random, so it is worth asking *how*:
    # a smooth, repetitive label stream would mean the model is leaning on
    # temporal persistence rather than on content.
    noise_labels = [s for s in ablated_predicted["noise"] if s.size]
    residual = residual_report(
        clean["scores"].argmax(axis=-1),
        ablated_collected["noise"]["scores"].argmax(axis=-1),
        clean["labels"],
        clean["mask"],
    )
    report["noise_behaviour"] = {
        "previous_frame_agreement": previous_frame_accuracy(noise_labels),
        "clean_previous_frame_agreement": previous_frame_accuracy(predicted_sequences),
        "pred_perplexity": perplexity(np.concatenate(noise_labels), num_labels)
        if noise_labels else 0.0,
        "residual": residual,
    }
    print(f"  noise behaviour: adjacent predicted frames agree "
          f"{report['noise_behaviour']['previous_frame_agreement']:.3f} of the time "
          f"(clean {report['noise_behaviour']['clean_previous_frame_agreement']:.3f}); "
          f"label perplexity {report['noise_behaviour']['pred_perplexity']:.1f}")
    print(f"  noise residual: {residual['ablated_accuracy']:.4f} accuracy, "
          f"{residual['accuracy_given_clean_correct']:.4f} where the clean pass was right "
          f"vs {residual['accuracy_given_clean_wrong']:.4f} where it was wrong "
          f"(lift={residual['lift']:+.4f}; {residual['hits_shared_with_clean']:.3f} of its "
          f"hits are shared with the clean pass, {residual['clean_accuracy']:.3f} = independence)")

    # ── 4. Causality + latency ──
    print("\n=== 4. Causality + latency ===")
    chunk_size = int(sce_cfg.get("chunk_size", 4))
    right_context = int(sce_cfg.get("right_context", 2))
    causality = causality_check(model, loader, chunk_size, right_context)
    report["causality"] = causality
    verdict = "PASS" if causality["prefix_unchanged"] else "FAIL"
    print(f"  causality [{verdict}]: frames < {causality['prefix_frames_checked']} are "
          f"unchanged by a rewrite of every frame from {causality['cut_frame']} onward "
          f"(max |Δ|={causality['prefix_max_delta']:.3e}); "
          f"last frame before the cut Δ={causality['last_frame_before_cut_delta']:.3e} "
          f"(non-zero expected: lookahead={causality['lookahead_frames']} frames)")

    timing = measure_rtf(model, loader, chunk_size)
    report["latency"] = timing
    print(f"  latency: {timing['latency_ms']:.1f} ms for {timing['frames']:.0f} frames "
          f"({timing['audio_seconds']:.2f}s audio) → "
          f"{timing['ms_per_frame']:.2f} ms/frame, RTF={timing['rtf']:.4f}")

    # ── 5. Checkpoint curve ──
    if args.extra_ckpt:
        print("\n=== 5. Checkpoint curve ===")
        curve = []
        for path in [args.ckpt] + list(args.extra_ckpt):
            if path == args.ckpt:
                row = metrics
            else:
                other = build_model(config, codebook)
                load_extractor_weights(other, path)
                other.eval()
                row, _, _ = core_metrics(other, loader, num_labels, args.ablation_seed)
            entry = {
                "checkpoint": path,
                "top1_utterance": row["top1_utterance"],
                "top1_frame": row["top1_frame"],
                "top5_frame": row["top5_frame"],
                "pred_perplexity": row["pred_perplexity"],
            }
            curve.append(entry)
            print(f"  {Path(path).name:24s} top1_utt={entry['top1_utterance']:.4f} "
                  f"top1_frame={entry['top1_frame']:.4f} top5={entry['top5_frame']:.4f} "
                  f"perp={entry['pred_perplexity']:.1f}")
        report["checkpoint_curve"] = curve

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  report written to {out_path}")


if __name__ == "__main__":
    main()
