"""Tests for the Stage-1 content-extractor evaluator (entry/eval_content_extractor.py).

Covered:
- ``valid_frames`` keeps exactly the masked frames.
- ``topk_accuracy`` reduces to plain accuracy at k=1 and is >= top-1 for k>1.
- ``previous_frame_accuracy`` and ``most_frequent_accuracy`` match hand-computed
  values on a small sequence, and the persistence baseline beats random on
  autocorrelated labels but not on shuffled ones.
- ``confusion_matrix`` / ``per_class_stats`` agree with a hand-built case.
- ``label_sequences`` strips the padded tail of each row.
- ``per_utterance_accuracy`` matches the quantity the trainer logs as val/acc (a
  mean over utterances) and differs from the frame-level average when lengths vary.
- ``residual_report`` tells a surviving-content ablation apart from one whose
  accuracy comes from a fallback output distribution, and ignores padded frames.
- ``bootstrap_ci`` is degenerate on a constant, brackets the mean, and is
  reproducible for a fixed seed.
- ``duration_bucket`` maps frame counts onto the requested intervals.
- ``align_scores`` crops and pads the prediction axis to the target length.
- The mel ablations (shuffle / zero / noise / reverse) touch only the valid
  region and leave the padded tail and the caller's tensor untouched.
- ``to_tensor`` / ``to_numpy`` round-trip numpy and paddle inputs.
- ``load_extractor_weights`` actually replaces the parameters (structured keys,
  ``extractor.`` prefix stripped) and raises on an architecture mismatch instead
  of silently evaluating randomly initialised weights.
"""

import sys
from pathlib import Path

import numpy as np
import paddle
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.eval_content_extractor import (
    align_scores,
    bootstrap_ci,
    confusion_matrix,
    duration_bucket,
    label_sequences,
    load_extractor_weights,
    make_ablation,
    most_frequent_accuracy,
    per_class_stats,
    per_utterance_accuracy,
    previous_frame_accuracy,
    residual_report,
    to_numpy,
    to_tensor,
    topk_accuracy,
    valid_frames,
)
from layers.stream_content_extractor import StreamContentExtractor


def test_valid_frames_keeps_only_masked_positions():
    scores = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    labels = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)

    kept_scores, kept_labels = valid_frames(scores, labels, mask)

    assert kept_scores.shape == (3, 4)
    assert kept_labels.tolist() == [1, 3, 5]
    # element [b, t, c] sits at b*12 + t*4 + c → (0,0)=0, (0,2)=8, (1,1)=16
    np.testing.assert_array_equal(kept_scores[:, 0], [0, 8, 16])


def test_topk_accuracy_at_k1_is_plain_accuracy_and_grows_with_k():
    scores = np.array([
        [5.0, 1.0, 0.0],   # target 0 → top-1 hit
        [0.0, 5.0, 1.0],   # target 1 → top-1 hit
        [1.0, 0.0, 5.0],   # target 2 → top-1 miss
    ], dtype=np.float32)
    target = np.array([0, 1, 1])

    assert topk_accuracy(scores, target, 1) == 2 / 3
    assert topk_accuracy(scores, target, 2) == 2 / 3
    assert topk_accuracy(scores, target, 3) == 1.0


def test_topk_accuracy_handles_empty_input():
    assert topk_accuracy(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64), 1) == 0.0


def test_previous_frame_baseline_matches_hand_computed_value():
    # 1,1,1,2,2,1 → adjacent pairs: ✓ ✓ ✗ ✓ ✗ → 3/5
    sequences = [np.array([1, 1, 1, 2, 2, 1], dtype=np.int64)]
    assert previous_frame_accuracy(sequences) == 3 / 5


def test_persistence_beats_random_on_autocorrelated_but_not_on_shuffled_labels():
    rng = np.random.default_rng(0)
    steady = np.repeat(np.arange(50, dtype=np.int64), 10)  # 500 frames, very sticky
    shuffled = rng.permutation(steady)

    assert previous_frame_accuracy([steady]) > 0.9
    assert previous_frame_accuracy([shuffled]) < 0.1


def test_most_frequent_baseline_is_the_majority_share():
    sequences = [np.array([0, 0, 0, 1, 2], dtype=np.int64)]
    assert most_frequent_accuracy(sequences, num_labels=4) == 3 / 5


def test_confusion_and_per_class_stats_on_hand_built_case():
    pred = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
    target = np.array([0, 1, 2, 2, 2, 0], dtype=np.int64)

    conf = confusion_matrix(pred, target, num_labels=3)

    # rows = target, cols = predicted
    np.testing.assert_array_equal(
        conf,
        np.array([[1, 0, 1],
                  [0, 1, 0],
                  [0, 1, 2]]),
    )
    support, accuracy = per_class_stats(conf)
    np.testing.assert_array_equal(support, [2, 1, 3])
    np.testing.assert_allclose(accuracy, [0.5, 1.0, 2 / 3])
    # classes with no support must report 0 accuracy, not NaN
    empty_conf = np.zeros((3, 3), dtype=np.int64)
    _, empty_acc = per_class_stats(empty_conf)
    assert np.isfinite(empty_acc).all()


def test_label_sequences_strips_padded_tail():
    labels = np.array([[7, 8, 9, 256, 256], [1, 2, 256, 256, 256]])
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=np.float32)

    sequences = label_sequences(labels, mask)

    assert [s.tolist() for s in sequences] == [[7, 8, 9], [1, 2]]


def test_per_utterance_accuracy_differs_from_the_frame_average():
    # utt 0: 3/3 correct; utt 1: 1/2 correct. Utterance mean 0.75, frame mean 0.8.
    scores = np.zeros((2, 4, 3), dtype=np.float32)
    scores[0, 0, 0] = scores[0, 1, 1] = scores[0, 2, 2] = 5.0
    scores[1, 0, 0] = scores[1, 1, 1] = 5.0
    labels = np.array([[0, 1, 2, 9], [0, 0, 9, 9]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float32)

    per_utterance = per_utterance_accuracy(scores, labels, mask)

    np.testing.assert_allclose(per_utterance, [1.0, 0.5])
    assert per_utterance.mean() == 0.75
    kept_scores, kept_labels = valid_frames(scores, labels, mask)
    assert (kept_scores.argmax(axis=1) == kept_labels).mean() == 0.8


def test_per_utterance_accuracy_skips_all_padding_rows():
    scores = np.zeros((2, 2, 2), dtype=np.float32)
    labels = np.zeros((2, 2), dtype=np.int64)
    mask = np.array([[1, 1], [0, 0]], dtype=np.float32)

    assert per_utterance_accuracy(scores, labels, mask).size == 1


def test_residual_report_separates_surviving_content_from_a_fallback():
    labels = np.full((1, 8), 3, dtype=np.int64)
    mask = np.ones((1, 8), dtype=np.float32)
    clean = np.array([[3, 3, 3, 3, 9, 9, 9, 9]])  # right on frames 0-3

    # Surviving content: the ablated hits are exactly the clean pass's successes.
    content = residual_report(
        clean_pred=clean,
        ablated_pred=np.array([[3, 3, 9, 9, 9, 9, 9, 9]]),
        labels=labels, mask=mask,
    )
    assert content["ablated_accuracy"] == 0.25
    assert content["accuracy_given_clean_correct"] == 0.5
    assert content["accuracy_given_clean_wrong"] == 0.0
    assert content["lift"] == 0.5
    assert content["hits_shared_with_clean"] == 1.0

    # Fallback distribution: equally likely to hit whether or not clean was right,
    # so the residual carries no information the clean pass did not already have.
    fallback = residual_report(
        clean_pred=clean,
        ablated_pred=np.array([[3, 9, 3, 9, 9, 3, 9, 3]]),
        labels=labels, mask=mask,
    )
    assert fallback["lift"] == 0.0
    assert fallback["hits_shared_with_clean"] == fallback["clean_accuracy"]


def test_residual_report_ignores_padded_frames():
    # The padded positions are "correct" for both passes and must not be counted.
    report = residual_report(
        clean_pred=np.array([[1, 1, 1, 1]]),
        ablated_pred=np.array([[9, 9, 9, 9]]),
        labels=np.array([[1, 1, 1, 1]]),
        mask=np.array([[1, 1, 0, 0]], dtype=np.float32),
    )

    assert report["frames"] == 2
    assert report["ablated_accuracy"] == 0.0
    assert report["lift"] == 0.0


def test_residual_report_is_zero_for_an_empty_mask():
    empty = np.zeros((0, 0), dtype=np.int64)

    report = residual_report(empty, empty, empty, np.zeros((0, 0), dtype=np.float32))

    assert report["frames"] == 0
    assert all(value == 0.0 for key, value in report.items() if key != "frames")


def test_bootstrap_ci_is_degenerate_for_a_constant_and_contains_the_mean():
    low, high = bootstrap_ci([0.5] * 40, reps=500, seed=0)
    assert low == high == 0.5

    values = np.linspace(0.0, 1.0, 40)
    low, high = bootstrap_ci(values, reps=500, seed=0)
    assert low < values.mean() < high
    assert (low, high) == bootstrap_ci(values, reps=500, seed=0)
    assert bootstrap_ci([], reps=10) == (0.0, 0.0)


def test_duration_bucket_maps_frame_counts_to_intervals():
    edges = (0, 100, 200, 400, 10 ** 9)

    buckets = duration_bucket([29, 100, 110, 250, 400, 1235], edges)

    assert buckets.tolist() == [0, 1, 1, 2, 3, 3]


def test_align_scores_crops_and_pads_prediction_axis():
    target = np.zeros((1, 5), dtype=np.int64)

    cropped = align_scores(np.arange(1 * 8 * 2, dtype=np.float32).reshape(1, 8, 2), target)
    assert cropped.shape == (1, 5, 2)

    padded = align_scores(np.ones((1, 3, 2), dtype=np.float32), target)
    assert padded.shape == (1, 5, 2)
    assert padded[:, 3:, :].sum() == 0.0

    exact = align_scores(np.ones((1, 5, 2), dtype=np.float32), target)
    assert exact.shape == (1, 5, 2)


def _mel_and_batch(length=6, total=10, n_mels=2):
    rng = np.random.default_rng(0)
    mel = paddle.to_tensor(rng.standard_normal((1, n_mels, total)).astype(np.float32))
    batch = {"lengths": np.array([length], dtype=np.int64)}
    return mel, batch


def test_ablation_zero_only_clears_the_valid_region():
    mel, batch = _mel_and_batch()
    original = to_numpy(mel).copy()

    out = to_numpy(make_ablation("zero")(mel, batch))

    assert np.abs(out[:, :, :6]).sum() == 0.0
    np.testing.assert_array_equal(out[:, :, 6:], original[:, :, 6:])
    np.testing.assert_array_equal(to_numpy(mel), original)  # caller's tensor untouched


def test_ablation_reverse_flips_only_the_valid_region():
    mel, batch = _mel_and_batch()
    original = to_numpy(mel).copy()

    out = to_numpy(make_ablation("reverse")(mel, batch))

    np.testing.assert_allclose(out[:, :, :6], original[:, :, :6][:, :, ::-1])
    np.testing.assert_array_equal(out[:, :, 6:], original[:, :, 6:])


def test_ablation_shuffle_is_a_permutation_within_the_valid_region():
    mel, batch = _mel_and_batch()
    original = to_numpy(mel).copy()

    out = to_numpy(make_ablation("shuffle", seed=3)(mel, batch))

    np.testing.assert_allclose(
        np.sort(out[:, :, :6], axis=2), np.sort(original[:, :, :6], axis=2)
    )
    np.testing.assert_array_equal(out[:, :, 6:], original[:, :, 6:])


def test_ablation_noise_preserves_shape_and_differs_from_input():
    mel, batch = _mel_and_batch()

    out = to_numpy(make_ablation("noise", seed=1)(mel, batch))

    assert out.shape == (1, 2, 10)
    assert not np.allclose(out[:, :, :6], to_numpy(mel)[:, :, :6])


def test_to_tensor_and_to_numpy_roundtrip():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = to_tensor(array)

    assert isinstance(tensor, paddle.Tensor)
    np.testing.assert_array_equal(to_numpy(tensor), array)
    np.testing.assert_array_equal(to_numpy(array), array)
    assert to_tensor(tensor) is tensor


def _tiny_extractor(num_labels=4):
    return StreamContentExtractor(
        input_dim=8, d_model=16, nhead=2, num_layers=1, output_dim=8,
        chunk_size=2, left_context=1, right_context=1, dim_feedforward=32,
        dropout=0.0, num_labels=num_labels,
    )


def test_load_extractor_weights_replaces_parameters(tmp_path):
    source = _tiny_extractor()
    ckpt_path = tmp_path / "tiny.pdparams"
    paddle.save(
        {"state_dict": {f"extractor.{k}": v for k, v in source.state_dict().items()}, "epoch": 3},
        str(ckpt_path),
    )

    target = _tiny_extractor()
    # Make the target obviously different first, so a no-op "load" cannot pass.
    target.mel_proj[0].weight.set_value(paddle.zeros_like(target.mel_proj[0].weight))

    load_extractor_weights(target, str(ckpt_path))

    source_state = source.state_dict()
    target_state = target.state_dict()
    for key in ("mel_proj.0.weight", "classifier.weight"):
        assert not np.allclose(to_numpy(target_state[key]), 0.0)
        np.testing.assert_array_equal(to_numpy(target_state[key]), to_numpy(source_state[key]))


def test_load_extractor_weights_raises_when_keys_do_not_match(tmp_path):
    # Reproduces the failure mode that made an earlier evaluation run report
    # chance-level accuracy as if it were a finding: keys in the wrong naming
    # scheme load *nothing*, and paddle only lists them as unexpected.
    ckpt_path = tmp_path / "mismatched.pdparams"
    paddle.save(
        {"state_dict": {"extractor.linear_0.w_0": paddle.zeros([4, 4])}},
        str(ckpt_path),
    )

    with pytest.raises(ValueError, match="does not match the model"):
        load_extractor_weights(_tiny_extractor(), str(ckpt_path))


if __name__ == "__main__":
    import inspect
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                if "tmp_path" in inspect.signature(fn).parameters:
                    fn(tmp_path=tmp)
                else:
                    fn()
                print(f"✓ {name}")
    print("\nAll tests passed!")
