"""Causal Mel Decoder — streaming mel-spectrogram generation.

Decodes mel-spectrogram frames from content + timbre + style + pitch
embeddings using causal convolutions.

Architecture:
    [z_c, z_t, z_s, f0] → Concat → CausalConv stack → Mel frames

Losses: MAE + SSIM + LSGAN adversarial (via MelDiscriminator).
"""

from typing import Optional

import paddle
import paddle.nn as nn

from layers.causal_conv import CausalConv1D


class CausalMelDecoder(nn.Layer):
    """Causal mel-spectrogram decoder.

    Args:
        content_dim: Content feature dimension.
        timbre_dim: Timbre embedding dimension.
        style_dim: Style embedding dimension.
        n_mels: Number of mel bins (default 80).
        hidden_dim: Hidden dimension.
        num_layers: Number of causal conv blocks.
        kernel_size: Causal conv kernel size.
    """

    def __init__(
        self,
        content_dim: int = 512,
        timbre_dim: int = 256,
        style_dim: int = 64,
        n_mels: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 5,
        kernel_size: int = 5,
    ):
        super().__init__()

        input_dim = content_dim + timbre_dim + style_dim + 1  # +1 for F0
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Sequential(
                    CausalConv1D(hidden_dim, hidden_dim, kernel_size, dilation=2 ** i),
                    nn.LeakyReLU(0.2),
                    nn.GroupNorm(1, hidden_dim),
                )
            )
        self.conv_layers = nn.LayerList(layers)

        self.output = nn.Sequential(
            CausalConv1D(hidden_dim, hidden_dim, 3),
            nn.LeakyReLU(0.2),
            CausalConv1D(hidden_dim, n_mels, 3),
        )

    def forward(
        self,
        z_c: paddle.Tensor,
        z_t: paddle.Tensor,
        z_s: paddle.Tensor,
        f0: paddle.Tensor,
    ) -> paddle.Tensor:
        """Generate mel-spectrogram from embeddings.

        Args:
            z_c: (B, T, content_dim) content embedding.
            z_t: (B, timbre_dim) global timbre embedding — will be broadcast.
            z_s: (B, T, style_dim) style embedding.
            f0: (B, T, 1) predicted F0.

        Returns:
            mel: (B, n_mels, T) predicted mel-spectrogram.
        """
        B, T = z_c.shape[0], z_c.shape[1]

        # Broadcast timbre across time
        z_t_b = z_t.unsqueeze(1).expand([-1, T, -1])  # (B, T, timbre_dim)

        # Concatenate all inputs
        x = paddle.concat([z_c, z_t_b, z_s, f0], axis=-1)  # (B, T, input_dim)
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Causal conv layers (operate on channel dim)
        x = x.transpose([0, 2, 1])  # (B, hidden_dim, T)
        for layer in self.conv_layers:
            identity = x
            x = layer(x)
            x = x + identity  # Residual
        x = self.output(x)  # (B, n_mels, T)

        return x
