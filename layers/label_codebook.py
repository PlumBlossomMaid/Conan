"""K-means clustering for producing discrete content labels from teacher features.

The Conan paper trains the Stream Content Extractor with a cross-entropy
loss against discrete content labels ("Softmax over J classes per frame,
selecting the highest-probability label"), whereas a regression objective
would need the teacher's continuous embedding. To follow the paper's
objective with pre-computed teacher features, we cluster the teacher
features offline into a J-entry codebook and distil each frame to the
nearest centroid index.

``PaddleLabelCodebook`` is intentionally tiny and dependency-free (no
sklearn, no paddle.cluster): the dataset is 375k x 256 float32, so a
k-means over a representative sample on CPU is a one-off few-minute job.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def kmeans(
    x: np.ndarray,
    num_clusters: int,
    max_iter: int = 100,
    n_init: int = 3,
    tol: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized k-means (Lloyd's algorithm, parallel across all points).

    Args:
        x: (N, D) float32 features.
        num_clusters: Number of centroids.
        max_iter: Maximum EM iterations.
        n_init: Number of restarts; the lowest-inertia run wins.
        tol: Relative inertia change below which a run is considered converged.
        rng: NumPy generator (deterministic when provided).
        verbose: Print per-iteration inertia.

    Returns:
        centroids: (num_clusters, D) float32.
        labels: (N,) int64 nearest centroid per point.
        inertia: (N,) float32 squared distance to the assigned centroid.
    """
    rng = rng or np.random.default_rng(0)
    n_points, dim = x.shape

    if num_clusters >= n_points:
        centroids = np.tile(x[:num_clusters], (1 + (n_points - 1) // num_clusters, 1))[:num_clusters]
        labels = np.arange(n_points) % num_clusters
        inertia = np.zeros(n_points, dtype=np.float32)
        return centroids, labels, inertia

    best_centroids: Optional[np.ndarray] = None
    best_inertia_total = float("inf")

    for init in range(n_init):
        seeds = x[rng.choice(n_points, size=num_clusters, replace=False)]
        centroids_run = seeds.astype(np.float32)

        for iteration in range(max_iter):
            # Batched pairwise distances via (a-b)^2 = |a|^2 - 2ab + |b|^2.
            dists = (
                (x * x).sum(axis=1, keepdims=True)
                - 2.0 * (x @ centroids_run.T)
                + (centroids_run * centroids_run).sum(axis=1, keepdims=True).T
            )
            labels_run = np.argmin(dists, axis=1)
            inertia_run = dists[np.arange(n_points), labels_run].astype(np.float32)

            totals = np.bincount(labels_run, minlength=num_clusters)
            # Vectorized centroid update via weighted bincount per feature:
            # O(N) per dimension and no (N, K) scratch matrix.
            sums = np.stack([
                np.bincount(labels_run, weights=x[:, dim], minlength=num_clusters)
                for dim in range(x.shape[1])
            ], axis=1).astype(np.float32)  # (num_clusters, D)
            new_centroids = np.where(
                totals[:, None] > 0,
                sums / np.maximum(totals[:, None], 1).astype(np.float32),
                centroids_run,
            )
            # Re-seed empty clusters to the point with the largest error,
            # which is the standard remedy for dead centroids.
            empty = np.where(totals == 0)[0]
            if empty.size:
                farthest = int(np.argmax(inertia_run))
                new_centroids[empty] = x[farthest].astype(np.float32)

            shift = float(np.mean(np.abs(new_centroids - centroids_run)))
            centroids_run = new_centroids
            if verbose:
                print(f"    kmeans init={init} iter={iteration} inertia={inertia_run.sum():.2f} shift={shift:.2e}", flush=True)
            if shift < tol:
                break

        inertia_total = float(inertia_run.sum())
        if verbose:
            print(f"    kmeans init={init} done, inertia={inertia_total:.2f}", flush=True)
        if inertia_total < best_inertia_total:
            best_inertia_total = inertia_total
            best_centroids = centroids_run

    return best_centroids, labels_run, inertia_run


class PaddleLabelCodebook:
    """Centroid table mapping continuous teacher features to label indices.

    Distances are Euclidean, so a lookup reduces to a single matrix product
    plus an axis-1 ``argmin`` (the ``||z||^2`` term is constant across
    centroids and cancels for a single query point).
    """

    def __init__(self, centroids: np.ndarray):
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[0] == 0:
            raise ValueError(f"centroids must be (J, D) with J > 0, got {centroids.shape}")
        self.centroids = centroids
        self.num_labels = centroids.shape[0]
        self.feature_dim = centroids.shape[1]

    def __len__(self) -> int:
        return self.num_labels

    def labels(self, features: np.ndarray) -> np.ndarray:
        """Nearest-centroid label per row.

        Args:
            features: (T, D) float32 teacher features.

        Returns:
            labels: (T,) int64.
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected (T, {self.feature_dim}), got {features.shape}"
            )
        logits = features @ self.centroids.T
        return np.argmax(logits, axis=1).astype(np.int64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.centroids)

    @staticmethod
    def load(path: str | Path) -> "PaddleLabelCodebook":
        return PaddleLabelCodebook(np.load(path))


def load_frame_sample(
    hdf5_paths: List[str],
    samples_per_file: int = 1,
    frames_per_file: Optional[int] = None,
    feature_name: str = "hubert",
    group_attr: str = "hubert_frames",
) -> np.ndarray:
    """Sample rows of teacher features from an HDF5 group-per-utterance file.

    Reads metadata only until enough frames are gathered, so a very small
    number of utterances is enough for clustering: the codebook covers the
    distribution of feature values, not the number of utterances.

    Args:
        hdf5_paths: Files to read, in order.
        samples_per_file: Frames drawn uniformly from each utterance.
        frames_per_file: Stop once this many frames are collected.
        feature_name: Dataset name holding the teacher features.
        group_attr: Group attribute giving the true feature frame count.

    Returns:
        (N, D) float32 frame matrix.
    """
    import h5py

    rows: List[np.ndarray] = []
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as h5:
            for group_name in sorted(h5.keys()):
                if frames_per_file is not None and sum(r.shape[0] for r in rows) >= frames_per_file:
                    return np.concatenate(rows, axis=0)
                group = h5[group_name]
                try:
                    frame_count = int(group.attrs[group_attr])
                except KeyError:
                    frame_count = group[feature_name].shape[0]
                if frame_count == 0:
                    continue
                features = group[feature_name][()]
                picks = np.linspace(0, frame_count - 1, num=min(samples_per_file, frame_count)).astype(np.int64)
                rows.append(features[picks])
    if not rows:
        raise ValueError("no teacher features found in the given HDF5 paths")
    return np.concatenate(rows, axis=0)


def build_codebook(
    features: np.ndarray,
    num_clusters: int,
    max_iter: int = 100,
    n_init: int = 3,
    seed: int = 0,
    verbose: bool = False,
) -> PaddleLabelCodebook:
    """Cluster ``features`` and wrap the centroids in a :class:`PaddleLabelCodebook`."""
    if features.shape[1] != 256:
        print(f"[info] clustering {features.shape[1]}-dim features (expected 256)", flush=True)
    rng = np.random.default_rng(seed)
    centroids, _, _ = kmeans(features, num_clusters, max_iter=max_iter, n_init=n_init, rng=rng, verbose=verbose)
    return PaddleLabelCodebook(centroids)


def perplexity(labels: np.ndarray, num_labels: int) -> float:
    """Codebook usage perplexity — a collapse diagnostic."""
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    proportions = counts / counts.sum()
    active = proportions > 0
    entropy = -float(np.sum(proportions[active] * np.log(proportions[active])))
    return math.exp(entropy)
