"""Tests for CVQ loss wiring in AdaptiveStyleEncoder and ConanMainModel.

Verifies that the CVQ codebook loss (VQ + commitment + contrastive) is
properly threaded through the training loop and contributes to the
generator loss.
"""

import sys
from pathlib import Path

import numpy as np
import paddle
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.adaptive_style_encoder import AdaptiveStyleEncoder
from layers.cvq import ClusteringVQ
from models.conan_main import ConanMainModel

# Run on CPU: the Iluvatar custom-device runtime raises SIGFPE at interpreter
# shutdown when GPU tensors are still alive (paddle custom-device teardown),
# which makes pytest exit 136 even though every test passes. These are pure
# logic tests; CPU keeps them hermetic and fast.
paddle.set_device("cpu")


def test_adaptive_style_encoder_returns_stats():
    """forward() must return (z_s, stats) where stats contains vq_loss."""
    paddle.seed(42)
    encoder = AdaptiveStyleEncoder(
        n_mels=80, style_dim=64, code_dim=64, num_codes=128,
        timbre_dim=256, content_dim=256,
    )
    encoder.eval()

    ref_mel = paddle.randn([2, 80, 40])
    z_c = paddle.randn([2, 20, 256])
    z_t = paddle.randn([2, 256])

    z_s, stats = encoder(ref_mel, z_c, z_t)

    assert z_s.shape == (2, 20, 64), f"Expected z_s shape (2, 20, 64), got {z_s.shape}"
    assert isinstance(stats, dict), f"Expected stats dict, got {type(stats)}"
    assert "vq_loss" in stats, f"Expected 'vq_loss' in stats, got {stats.keys()}"
    assert paddle.is_tensor(stats["vq_loss"]), f"Expected vq_loss to be a tensor"
    assert float(stats["vq_loss"]) > 0, "vq_loss should be positive"


def test_extract_style_returns_single_tensor():
    """extract_style() must return only z_s (no stats) for inference."""
    paddle.seed(42)
    encoder = AdaptiveStyleEncoder(
        n_mels=80, style_dim=64, code_dim=64, num_codes=128,
        timbre_dim=256, content_dim=256,
    )
    encoder.eval()

    ref_mel = paddle.randn([2, 80, 40])
    z_c = paddle.randn([2, 20, 256])
    z_t = paddle.randn([2, 256])

    z_s = encoder.extract_style(ref_mel, z_c, z_t)

    assert z_s.shape == (2, 20, 64)
    assert paddle.is_tensor(z_s), f"Expected tensor, got {type(z_s)}"


def test_cvq_contrastive_loss_vectorized():
    """ClusteringVQ._compute_contrastive_loss must handle n_pos=1 without shape error."""
    paddle.seed(42)
    cvq = ClusteringVQ(code_dim=64, num_codes=128, beta=0.25, contrastive_weight=0.1)

    # Test with batch size 1 (n_pos=1 for each code)
    z = paddle.randn([1, 64])
    codes = paddle.to_tensor([0])  # Single sample assigned to code 0

    loss = cvq._compute_contrastive_loss(z, codes)

    assert paddle.is_tensor(loss)
    assert loss.shape == [], f"Expected scalar loss, got shape {loss.shape}"
    assert float(loss) >= 0, "Contrastive loss should be non-negative"


def test_cvq_contrastive_loss_multiple_samples():
    """Contrastive loss with multiple samples per code."""
    paddle.seed(42)
    cvq = ClusteringVQ(code_dim=64, num_codes=128, beta=0.25, contrastive_weight=0.1)

    # Multiple samples, some codes have n_pos > 1
    z = paddle.randn([10, 64])
    codes = paddle.to_tensor([0, 0, 0, 1, 1, 2, 3, 4, 5, 6])

    loss = cvq._compute_contrastive_loss(z, codes)

    assert paddle.is_tensor(loss)
    assert loss.shape == []
    assert float(loss) >= 0


def test_conan_main_model_vq_loss_in_forward():
    """ConanMainModel.forward() must return vq_loss in the output dict."""
    paddle.seed(42)
    config = {
        "audio": {"num_mels": 80},
        "main_model": {},
        "loss": {},
        "training": {"accumulate_grad_batches": 1},
    }
    model = ConanMainModel(config, content_extractor=None)
    model.eval()

    src_mel = paddle.randn([2, 80, 20])
    ref_mel = paddle.randn([2, 80, 40])
    f0 = paddle.randn([2, 20, 1])

    out = model.forward(src_mel, ref_mel, f0)

    assert "vq_loss" in out, f"Expected 'vq_loss' in output, got {out.keys()}"
    assert paddle.is_tensor(out["vq_loss"])
    assert float(out["vq_loss"]) > 0, "vq_loss should be positive"


def test_conan_main_model_generator_loss_includes_vq():
    """_generator_loss() must include loss_vq in the returned dict."""
    paddle.seed(42)
    config = {
        "audio": {"num_mels": 80},
        "main_model": {},
        "loss": {},
        "training": {"accumulate_grad_batches": 1},
    }
    model = ConanMainModel(config, content_extractor=None)
    model.eval()

    src_mel = paddle.randn([2, 80, 20])
    ref_mel = paddle.randn([2, 80, 40])
    f0 = paddle.randn([2, 20, 1])

    out = model.forward(src_mel, ref_mel, f0)

    g_losses = model._generator_loss(
        out["mel_pred"], src_mel, out["f0_pred"], f0,
        vq_loss=out.get("vq_loss")
    )

    assert "loss_vq" in g_losses, f"Expected 'loss_vq' in losses, got {g_losses.keys()}"
    assert paddle.is_tensor(g_losses["loss_vq"])
    assert float(g_losses["loss_vq"]) > 0, "loss_vq should be positive"
    assert float(g_losses["loss_g"]) >= float(g_losses["loss_vq"]), \
        "loss_g should include loss_vq contribution"


def test_lambda_vq_configurable():
    """lambda_vq must scale the VQ contribution in the generator loss.

    Uses a single model instance and rewrites ``lambda_vq`` in place so the
    test does not allocate a second full model (six full models in one
    pytest process can OOM/crash the runner on smaller memory budgets).
    """
    paddle.seed(42)
    config = {
        "audio": {"num_mels": 80},
        "main_model": {},
        "loss": {},
        "training": {"accumulate_grad_batches": 1},
    }
    model = ConanMainModel(config, content_extractor=None)
    model.eval()

    src_mel = paddle.randn([2, 80, 20])
    ref_mel = paddle.randn([2, 80, 40])
    f0 = paddle.randn([2, 20, 1])

    out = model.forward(src_mel, ref_mel, f0)
    mel_pred, f0_pred = out["mel_pred"], out["f0_pred"]
    vq_loss = out.get("vq_loss")
    assert float(vq_loss) > 0

    model.lambda_vq = 1.0
    losses_full = model._generator_loss(
        mel_pred, src_mel, f0_pred, f0, vq_loss=vq_loss
    )
    loss_vq_full = float(losses_full["loss_vq"])
    total_full = float(losses_full["loss_g"])

    model.lambda_vq = 2.0
    losses_doubled = model._generator_loss(
        mel_pred, src_mel, f0_pred, f0, vq_loss=vq_loss
    )
    assert float(losses_doubled["loss_vq"]) == pytest.approx(
        2.0 * loss_vq_full, rel=1e-6
    )

    model.lambda_vq = 0.0
    losses_zero = model._generator_loss(
        mel_pred, src_mel, f0_pred, f0, vq_loss=vq_loss
    )
    assert float(losses_zero["loss_vq"]) == 0.0
    # With the VQ term disabled the total must drop by exactly that amount.
    assert float(losses_zero["loss_g"]) == pytest.approx(
        total_full - loss_vq_full, rel=1e-6
    )


if __name__ == "__main__":
    test_adaptive_style_encoder_returns_stats()
    print("✓ test_adaptive_style_encoder_returns_stats")
    test_extract_style_returns_single_tensor()
    print("✓ test_extract_style_returns_single_tensor")
    test_cvq_contrastive_loss_vectorized()
    print("✓ test_cvq_contrastive_loss_vectorized")
    test_cvq_contrastive_loss_multiple_samples()
    print("✓ test_cvq_contrastive_loss_multiple_samples")
    test_conan_main_model_vq_loss_in_forward()
    print("✓ test_conan_main_model_vq_loss_in_forward")
    test_conan_main_model_generator_loss_includes_vq()
    print("✓ test_conan_main_model_generator_loss_includes_vq")
    test_lambda_vq_configurable()
    print("✓ test_lambda_vq_configurable")
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_adaptive_style_encoder_returns_stats()
    print("✓ test_adaptive_style_encoder_returns_stats")
    test_extract_style_returns_single_tensor()
    print("✓ test_extract_style_returns_single_tensor")
    test_cvq_contrastive_loss_vectorized()
    print("✓ test_cvq_contrastive_loss_vectorized")
    test_cvq_contrastive_loss_multiple_samples()
    print("✓ test_cvq_contrastive_loss_multiple_samples")
    test_conan_main_model_vq_loss_in_forward()
    print("✓ test_conan_main_model_vq_loss_in_forward")
    test_conan_main_model_generator_loss_includes_vq()
    print("✓ test_conan_main_model_generator_loss_includes_vq")
    test_lambda_vq_configurable()
    print("✓ test_lambda_vq_configurable")
    print("\nAll tests passed!")
