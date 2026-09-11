"""Tests for CVQ loss wiring in AdaptiveStyleEncoder and ConanMainModel.

Verifies that the CVQ codebook loss (VQ + commitment + contrastive) is
properly threaded through the training loop and contributes to the
generator loss.
"""

import sys
from pathlib import Path

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.adaptive_style_encoder import AdaptiveStyleEncoder
from layers.cvq import ClusteringVQ
from models.conan_main import ConanMainModel


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
    """lambda_vq must be configurable via the loss config."""
    paddle.seed(42)

    # Test with default lambda_vq=1.0
    config_default = {
        "audio": {"num_mels": 80},
        "main_model": {},
        "loss": {},
        "training": {"accumulate_grad_batches": 1},
    }
    model_default = ConanMainModel(config_default, content_extractor=None)
    model_default.eval()

    src_mel = paddle.randn([2, 80, 20])
    ref_mel = paddle.randn([2, 80, 40])
    f0 = paddle.randn([2, 20, 1])

    out = model_default.forward(src_mel, ref_mel, f0)
    vq_loss_raw = float(out["vq_loss"])

    g_losses_default = model_default._generator_loss(
        out["mel_pred"], src_mel, out["f0_pred"], f0,
        vq_loss=out.get("vq_loss")
    )
    loss_vq_default = float(g_losses_default["loss_vq"])

    # Test with lambda_vq=0.0 (should zero out the loss)
    config_zero = {
        "audio": {"num_mels": 80},
        "main_model": {},
        "loss": {"vq_weight": 0.0},
        "training": {"accumulate_grad_batches": 1},
    }
    model_zero = ConanMainModel(config_zero, content_extractor=None)
    model_zero.eval()

    out_zero = model_zero.forward(src_mel, ref_mel, f0)
    g_losses_zero = model_zero._generator_loss(
        out_zero["mel_pred"], src_mel, out_zero["f0_pred"], f0,
        vq_loss=out_zero.get("vq_loss")
    )
    loss_vq_zero = float(g_losses_zero["loss_vq"])

    assert loss_vq_default > 0, "Default lambda_vq should produce positive loss"
    assert loss_vq_zero == 0.0, f"lambda_vq=0.0 should zero out loss_vq, got {loss_vq_zero}"


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
