import sys
from pathlib import Path

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.preprocess_hubert import _run_hubert_batch
from layers.hubert import HubertTeacher, hubert_frame_count


def test_hubert_frame_count_matches_feature_extractor():
    assert hubert_frame_count(640) == 1
    assert hubert_frame_count(1280) == 3


def test_hubert_encoder_masks_padded_features():
    paddle.seed(1234)
    teacher = HubertTeacher()
    teacher.eval()
    features = paddle.randn([2, 3, 512])
    padding_mask = paddle.to_tensor([[False, True, True], [False, False, False]])

    with paddle.no_grad():
        actual = teacher.encode_features(features, padding_mask=padding_mask).numpy()

    assert actual.shape == (2, 3, 256)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[0, 1:], 0.0, atol=1e-6)


def test_run_hubert_batch_preserves_individual_waveform_lengths():
    class RecordingModel:
        def __init__(self):
            self.shapes = []

        def feature_extractor(self, source):
            self.shapes.append(tuple(source.shape))
            return paddle.zeros([source.shape[0], 512, source.shape[-1] // 320])

        def encode_features(self, features, padding_mask=None):
            assert padding_mask is not None
            return paddle.zeros([features.shape[0], features.shape[1], 256])

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
