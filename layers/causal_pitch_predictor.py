"""Causal Pitch Predictor — streaming F0 estimation from content features.

Predicts F0 from the content embedding using causal convolutions.

Loss: MSE between predicted and ground truth F0 (from WORLD or similar).
"""

import paddle
import paddle.nn as nn

from layers.causal_conv import CausalConv1D


class CausalPitchPredictor(nn.Layer):
    """Causal pitch predictor.

    Predicts F0 values frame by frame from content features,
    using only causal context (no future frames).

    Architecture:
        z_c → [CausalConv1D × N] → Linear → F0

    Args:
        content_dim: Content feature dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of causal conv layers.
        kernel_size: Conv kernel size.
    """

    def __init__(
        self,
        content_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(content_dim, hidden_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    CausalConv1D(hidden_dim, hidden_dim, kernel_size, dilation=1),
                    nn.LeakyReLU(0.2),
                    nn.GroupNorm(1, hidden_dim),  # channel-wise norm (works with conv output)
                )
            )
        self.conv_layers = nn.LayerList(layers)

        self.output = nn.Linear(hidden_dim, 1)  # Single F0 value per frame

    def forward(self, z_c: paddle.Tensor) -> paddle.Tensor:
        """Predict F0 from content features.

        Args:
            z_c: (B, T, content_dim) content embedding.

        Returns:
            f0: (B, T, 1) predicted F0 values.
        """
        x = self.input_proj(z_c)  # (B, T, hidden_dim)

        # Causal conv layers (operate on channel dimension)
        x = x.transpose([0, 2, 1])  # (B, hidden_dim, T)
        for layer in self.conv_layers:
            x = layer(x)  # (B, hidden_dim, T)
        x = x.transpose([0, 2, 1])  # (B, T, hidden_dim)

        f0 = self.output(x)  # (B, T, 1)
        return f0
