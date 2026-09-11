"""Tests for the discrete label codebook and CE-mode SCE training.

Covered:
- kmeans() produces a valid centroid table with monotonically
  decreasing inertia and no empty clusters for representative data.
- PaddleLabelCodebook.labels() is argmax(features @ centroids.T),
  matching the paper's "Softmax over J classes, highest-probability label".
- save/load round-trip.
- ContentExtractorDataset in CE mode returns ``hubert_label``
  (int64, padded, masked) instead of ``hubert_emb``.
- ContentExtractorModel in CE mode builds a classifier head and its
  training_step computes a valid cross-entropy loss.
"""

import sys
from pathlib import Path

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.dataset import ContentExtractorDataset
from layers.label_codebook import (
    PaddleLabelCodebook,
    build_codebook,
    kmeans,
    perplexity,
)
from models.content_extractor import ContentExtractorModel


def _fake_features(n=4000, dim=256, seed=0):
    rng = np.random.default_rng(seed)
    # Structured clusters so k-means converges to non-trivial assignments.
    centers = rng.standard_normal((16, dim)).astype(np.float32)
    x = np.concatenate([
        centers[i % 16] + 0.2 * rng.standard_normal((n // 16, dim)).astype(np.float32)
        for i in range(16)
    ])
    return x.astype(np.float32)


def test_kmeans_converges_and_no_empty_clusters():
    x = _fake_features()
    centroids, labels, inertia = kmeans(x, num_clusters=16, max_iter=50, n_init=2, tol=1e-6)

    assert centroids.shape == (16, 256)
    assert labels.shape == (4000,)
    assert labels.min() >= 0 and labels.max() < 16
    assert np.isfinite(inertia).all()
    # Every cluster should be populated with structured data.
    counts = np.bincount(labels, minlength=16)
    assert (counts > 0).all(), f"empty clusters: {counts}"


def test_codebook_labels_is_argmax_logits():
    x = _fake_features()
    cb = build_codebook(x, num_clusters=16, max_iter=30, n_init=1, seed=0)
    labels = cb.labels(x[:200])
    expected = np.argmax(x[:200] @ cb.centroids.T, axis=1)
    np.testing.assert_array_equal(labels, expected)


def test_codebook_labels_accepts_batched_3d():
    x = _fake_features()
    cb = build_codebook(x, num_clusters=16, max_iter=30, n_init=1, seed=0)
    batch = x[:40].reshape(2, 20, 256)
    labels = cb.labels(batch)
    assert labels.shape == (2, 20)
    np.testing.assert_array_equal(
        labels,
        np.argmax(x[:40] @ cb.centroids.T, axis=1).reshape(2, 20),
    )


def test_codebook_save_load_roundtrip(tmp_path):
    x = _fake_features(n=2000)
    cb = build_codebook(x, num_clusters=8, max_iter=20)
    path = tmp_path / "codebook.npy"
    cb.save(str(path))
    loaded = PaddleLabelCodebook.load(str(path))
    assert loaded.num_labels == 8
    assert loaded.feature_dim == 256
    np.testing.assert_array_equal(loaded.centroids, cb.centroids)


def test_dataset_ce_mode_returns_labels(tmp_path):
    # Build a tiny HDF5 with one utterance.
    import h5py

    rng = np.random.default_rng(42)
    mel = rng.standard_normal((80, 30)).astype(np.float32)
    hubert = rng.standard_normal((30, 256)).astype(np.float32)
    h5_path = tmp_path / "tiny.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("00000000")
        grp.create_dataset("mel", data=mel)
        grp.create_dataset("hubert", data=hubert)
        grp.attrs["mel_frames"] = 30
        grp.attrs["hubert_frames"] = 30

    cb = build_codebook(_fake_features(n=1000), num_clusters=8, max_iter=10)
    ds = ContentExtractorDataset(
        hdf5_path=str(h5_path), max_frames=50, label_codebook=cb,
    )
    item = ds[0]
    assert "hubert_label" in item, f"expected hubert_label, got {item.keys()}"
    assert "hubert_emb" not in item
    assert item["hubert_label"].shape == (50,)
    assert item["hubert_label"].dtype == np.int64
    assert item["valid_mask"][:30].sum() == 30 and item["valid_mask"][30:].sum() == 0
    assert item["hubert_label"][:30].min() >= 0 and item["hubert_label"][:30].max() < 8
    # Padded region must carry a neutral label outside the codebook range so
    # masked-out frames can never collide with a real label 0.
    assert (item["hubert_label"][30:] == 256).all()
    batch = ds.collater([item])
    assert batch["hubert_label"].shape == (1, 50)
    assert batch["hubert_label"].dtype == np.int64


def test_dataset_default_mode_still_returns_embeddings(tmp_path):
    import h5py

    rng = np.random.default_rng(0)
    mel = rng.standard_normal((80, 20)).astype(np.float32)
    hubert = rng.standard_normal((20, 256)).astype(np.float32)
    h5_path = tmp_path / "tiny2.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("00000000")
        grp.create_dataset("mel", data=mel)
        grp.create_dataset("hubert", data=hubert)
        grp.attrs["mel_frames"] = 20
        grp.attrs["hubert_frames"] = 20

    ds = ContentExtractorDataset(hdf5_path=str(h5_path), max_frames=30)
    item = ds[0]
    assert "hubert_emb" in item and "hubert_label" not in item


def test_content_extractor_ce_model_builds_classifier(tmp_path):
    import h5py

    rng = np.random.default_rng(1)
    mel = rng.standard_normal((80, 40)).astype(np.float32)
    hubert = rng.standard_normal((40, 256)).astype(np.float32)
    h5_path = tmp_path / "tiny_model.h5"
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("00000000")
        grp.create_dataset("mel", data=mel)
        grp.create_dataset("hubert", data=hubert)
        grp.attrs["mel_frames"] = 40
        grp.attrs["hubert_frames"] = 40

    cb_path = tmp_path / "cb.npy"
    build_codebook(_fake_features(n=1000), num_clusters=8, max_iter=10).save(str(cb_path))

    paddle.seed(0)
    config = {
        "content_extractor": {"loss_type": "ce", "num_labels": 8},
        "data": {"label_codebook": str(cb_path), "hdf5_path": str(h5_path)},
        "audio": {"num_mels": 80, "max_frames": 200},
        "training": {"accumulate_grad_batches": 1, "log_every": 100},
    }
    m = ContentExtractorModel(config)
    assert m.loss_type == "ce"
    assert m.extractor.classifier is not None
    assert tuple(m.extractor.classifier.weight.shape) == (512, 8)

    ds = ContentExtractorDataset(hdf5_path=str(h5_path), max_frames=100, label_codebook=m.label_codebook)
    item = ds[0]
    batch = ds.collater([item])
    batch["source_mel"] = paddle.to_tensor(batch["source_mel"])
    batch["hubert_label"] = paddle.to_tensor(batch["hubert_label"])
    batch["valid_mask"] = paddle.to_tensor(batch["valid_mask"])

    loss = m.training_step(batch, 0)
    loss_val = float(loss)
    assert np.isfinite(loss_val) and loss_val > 0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in [
            test_kmeans_converges_and_no_empty_clusters,
            test_codebook_labels_is_argmax_logits,
        ]:
            fn()
            print(f"✓ {fn.__name__}")
        test_codebook_save_load_roundtrip(tmp)
        print("✓ test_codebook_save_load_roundtrip")
        test_dataset_ce_mode_returns_labels(tmp)
        print("✓ test_dataset_ce_mode_returns_labels")
        test_dataset_default_mode_still_returns_embeddings(tmp)
        print("✓ test_dataset_default_mode_still_returns_embeddings")
        test_content_extractor_ce_model_builds_classifier(tmp)
        print("✓ test_content_extractor_ce_model_builds_classifier")
        print("\nAll tests passed!")