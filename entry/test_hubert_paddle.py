import sys
from pathlib import Path

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.preprocess_hubert import _run_hubert_batch
from layers.hubert import HubertTeacher


def test_run_hubert_batch_preserves_individual_waveform_lengths():
    class RecordingModel:
        def __init__(self):
            self.shapes = []

        def __call__(self, source):
            self.shapes.append(tuple(source.shape))
            return paddle.zeros([1, source.shape[-1] // 320, 256])

    model = RecordingModel()
    samples = [
        {"audio": np.zeros(640, dtype=np.float32)},
        {"audio": np.zeros(1280, dtype=np.float32)},
    ]

    embeddings = _run_hubert_batch(model, samples)

    assert model.shapes == [(1, 1, 640), (1, 1, 1280)]
    assert [embedding.shape for embedding in embeddings] == [(2, 256), (4, 256)]


def test_hubert_checkpoint_loads_without_key_mismatches():
    checkpoint_path = Path(r"E:\code\1\hubert4_paddle_aligned_20260818.pdparams")
    if not checkpoint_path.exists():
        return

    teacher = HubertTeacher()
    teacher.load_pretrained(checkpoint_path)
