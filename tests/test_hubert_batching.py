from pathlib import Path

import numpy as np

from entry.preprocess_hubert import _batches
from layers.batching import batch_by_files


def test_batch_by_files_respects_padded_frame_and_attention_budgets():
    lengths = {"short": 500, "medium": 1000, "long": 2000}
    batches = batch_by_files(
        ["short", "medium", "long"],
        lengths.__getitem__,
        max_batch_size=8,
        max_batch_frames=4000,
        max_attention_tokens=2_500_000,
    )

    assert batches == [["short", "medium"], ["long"]]
    for batch in batches:
        padded_length = max(lengths[item] for item in batch)
        if padded_length <= 1_500:
            assert len(batch) * padded_length <= 4000
            assert len(batch) * padded_length**2 <= 2_500_000


def test_batch_by_files_keeps_an_over_budget_item_intact():
    batches = batch_by_files(
        ["long"],
        lambda _: 5000,
        max_batch_size=8,
        max_batch_frames=4000,
        max_attention_tokens=1_000_000,
    )

    assert batches == [["long"]]


def test_preprocess_batches_sort_paths_by_audio_length(tmp_path):
    import soundfile as sf

    paths = []
    for name, samples in (("long", 16000), ("short", 3200), ("medium", 9600)):
        path = tmp_path / f"{name}.wav"
        sf.write(path, np.zeros(samples, dtype=np.float32), 16000)
        paths.append(path)

    batches = _batches(
        paths,
        max_batch_size=2,
        max_batch_frames=1000,
        max_attention_tokens=200_000,
        sample_rate=16000,
        hop_size=320,
    )

    ordered = [path.stem for batch in batches for path in batch]
    assert ordered == ["short", "medium", "long"]
