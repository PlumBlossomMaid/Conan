"""``forward_chunk`` must reproduce ``forward`` on the same input.

Training and offline evaluation go through ``EmformerEncoder.forward``, which
walks the chunks in a loop and hands every layer the *previous chunk's* output as
its memory bank — the memory is only reassigned between chunks. A streaming
implementation that reassigns the memory inside the layer loop therefore feeds
each layer a different bank than the one the model was trained with, and the
resulting outputs are not a drop-in replacement: measured on the trained
checkpoint the streaming replay agreed with the batch path on only 37% of frames.

The replay driven here is the same caller ``entry/verify_streaming.py`` uses, so
this test fails if either the caller's context construction or the module's
streaming loop drifts from the batch semantics.
"""

import sys
from pathlib import Path

import numpy as np
import paddle
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.eval_content_extractor import to_numpy  # noqa: E402
from entry.verify_streaming import streaming_forward  # noqa: E402
from layers.stream_content_extractor import StreamContentExtractor  # noqa: E402


def _tiny_extractor(chunk_size=4, left_context=1, right_context=2, num_layers=3):
    return StreamContentExtractor(
        input_dim=8, d_model=16, nhead=2, num_layers=num_layers, output_dim=8,
        chunk_size=chunk_size, left_context=left_context, right_context=right_context,
        dim_feedforward=32, dropout=0.0, num_labels=4,
    )


@pytest.mark.parametrize("length", [16, 25, 39])
def test_streaming_replay_matches_the_batch_forward(length):
    paddle.seed(0)
    model = _tiny_extractor()
    model.eval()
    mel = paddle.to_tensor(np.random.default_rng(0).standard_normal((1, length, 8)).astype("float32"))

    batch_out = to_numpy(model(mel))
    replayed = to_numpy(streaming_forward(model, mel))

    assert replayed.shape == batch_out.shape
    np.testing.assert_allclose(replayed, batch_out, atol=1e-5)


def test_streaming_replay_keeps_the_memory_bank_fixed_across_layers():
    # The discriminating case: with num_layers > 1 the buggy loop feeds layer k
    # the output of layer k-1 instead of the previous chunk. One layer cannot
    # tell the two apart, so a single-layer model would hide the defect.
    paddle.seed(0)
    model = _tiny_extractor(num_layers=1)
    model.eval()
    mel = paddle.to_tensor(np.random.default_rng(1).standard_normal((1, 25, 8)).astype("float32"))

    np.testing.assert_allclose(
        to_numpy(streaming_forward(model, mel)), to_numpy(model(mel)), atol=1e-5
    )


def test_streaming_replay_respects_the_right_context():
    # Rewriting the chunk after the right-context window must not change a chunk
    # whose own window has already closed.
    paddle.seed(0)
    model = _tiny_extractor()
    model.eval()
    mel = paddle.to_tensor(np.random.default_rng(2).standard_normal((1, 40, 8)).astype("float32"))

    clean = to_numpy(streaming_forward(model, mel))
    perturbed = to_numpy(streaming_forward(model, mel, perturb_from_chunk=8))

    # chunk j depends on chunks [j-1, j+right_context] = [j-1, j+2] → j <= 5 is safe
    boundaries = (5 + 1) * model.emformer.chunk_size
    np.testing.assert_allclose(perturbed[:, :boundaries], clean[:, :boundaries], atol=1e-5)
    assert not np.allclose(perturbed[:, boundaries:], clean[:, boundaries:], atol=1e-5)


def test_the_summary_cannot_reach_the_chunk_output():
    # A summary only enters the attention queries and query rows do not attend to
    # each other, so it reaches ``new_summary`` but never the returned chunk.
    # Measured against two very different summaries: the mean of the incoming
    # chunk (what ``forward`` computes) and the layer's own ``new_summary`` (what
    # a streaming caller would have lying around). Streaming callers may pass
    # zeros; this is why.
    paddle.seed(0)
    model = _tiny_extractor()
    model.eval()
    mel = paddle.to_tensor(np.random.default_rng(3).standard_normal((1, 40, 8)).astype("float32"))

    recomputed = to_numpy(streaming_forward(model, mel, carry_summary=False))
    carried = to_numpy(streaming_forward(model, mel, carry_summary=True))

    np.testing.assert_allclose(recomputed, carried, atol=1e-6)


if __name__ == "__main__":
    import inspect
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, fn in sorted(globals().items()):
            if not (name.startswith("test_") and callable(fn)):
                continue
            required = [
                parameter.name
                for parameter in inspect.signature(fn).parameters.values()
                if parameter.default is inspect.Parameter.empty
            ]
            if required not in ([], ["tmp_path"]):
                continue  # parametrised; pytest supplies the argument
            fn(tmp_path=tmp) if required else fn()
            print(f"✓ {name}")
    print("\nAll tests passed!")
