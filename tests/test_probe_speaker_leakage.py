"""Tests for entry/probe_speaker_leakage.py.

Covered:
- ``stratified_folds`` balances every class across folds and is deterministic.
- ``fit_softmax`` / ``softmax_predict`` separate a linearly separable problem.
- ``probe_cv_accuracy`` sits at chance when the features carry no label
  information, and is high when they encode the label directly — the two ends the
  real probe's controls rely on.
- ``label_histogram`` normalises per utterance and is all-zero for empty input.
- ``majority_share`` returns the largest class share.
- ``held_out_overlap`` matches by basename, so a probe set drawn from the train
  split reports 0 held out rather than silently claiming otherwise.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.probe_speaker_leakage import (
    fit_softmax,
    held_out_overlap,
    label_histogram,
    majority_share,
    probe_cv_accuracy,
    softmax_predict,
    stratified_folds,
)


def test_held_out_overlap_matches_by_basename():
    probe = ["/probe/100_1_000001_000000.wav", "/probe/200_2_000002_000000.wav"]

    report = held_out_overlap(probe, ["/elsewhere/200_2_000002_000000.wav"])

    assert report["probe_utterances"] == 2
    assert report["probe_utterances_held_out"] == 1
    assert report["held_out_fraction"] == 0.5


def test_held_out_overlap_is_zero_for_a_disjoint_train_drawn_set():
    probe = ["/probe/100_1_000001_000000.wav", "/probe/100_1_000002_000000.wav"]

    report = held_out_overlap(probe, ["/valid/999_9_000003_000000.wav"])

    assert report["probe_utterances_held_out"] == 0
    assert report["held_out_fraction"] == 0.0


def test_held_out_overlap_is_zero_for_an_empty_probe_set():
    report = held_out_overlap([], ["/valid/999_9_000003_000000.wav"])

    assert report["probe_utterances"] == 0
    assert report["held_out_fraction"] == 0.0


def test_stratified_folds_balances_each_class():
    targets = np.repeat(np.arange(4), 8)  # 4 classes x 8 items

    assignment = stratified_folds(targets, folds=4, seed=0)

    for label in range(4):
        counts = np.bincount(assignment[targets == label], minlength=4)
        assert counts.tolist() == [2, 2, 2, 2]
    np.testing.assert_array_equal(assignment, stratified_folds(targets, 4, 0))


def test_fit_softmax_separates_a_linearly_separable_problem():
    features = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.1], [0.1, 3.0]], dtype=np.float64)
    targets = np.array([0, 1, 0, 1])

    weights = fit_softmax(features, targets, iters=800)

    np.testing.assert_array_equal(softmax_predict(features, weights), targets)
    assert softmax_predict(np.array([[9.0, 0.0]]), weights)[0] == 0
    assert softmax_predict(np.array([[0.0, 9.0]]), weights)[0] == 1


def test_probe_cv_accuracy_is_at_chance_for_uninformative_features():
    rng = np.random.default_rng(0)
    targets = np.repeat(np.arange(8), 25)  # 200 items, 8 classes -> chance 0.125
    features = rng.normal(size=(targets.size, 12))

    accuracy = probe_cv_accuracy(features, targets, folds=4, seed=0)

    assert abs(accuracy - 1 / 8) < 0.12


def test_probe_cv_accuracy_is_high_when_features_encode_the_class():
    rng = np.random.default_rng(1)
    targets = np.repeat(np.arange(8), 25)
    features = np.zeros((targets.size, 8))
    features[np.arange(targets.size), targets] = 1.0
    features += rng.normal(scale=0.01, size=features.shape)

    accuracy = probe_cv_accuracy(features, targets, folds=4, seed=0)

    assert accuracy > 0.95


def test_label_histogram_normalises_and_handles_empty_sequences():
    sequences = [np.array([0, 0, 1, 1], dtype=np.int64), np.array([], dtype=np.int64)]

    histograms = label_histogram(sequences, num_labels=3)

    np.testing.assert_allclose(histograms[0], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(histograms[1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(histograms.sum(axis=1), [1.0, 0.0])


def test_majority_share_is_the_largest_class_share():
    assert majority_share([1, 1, 1, 2, 2]) == 3 / 5
    assert majority_share([]) == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"✓ {name}")
    print("\nAll tests passed!")
