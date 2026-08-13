"""Timbre Encoder — convolutional speaker embedding extractor.

Extracts a global timbre embedding z_t from the reference mel-spectrogram
using a stack of causal convolutional blocks with downsampling.

Architecture:
    Ref Mel → [ConvBlock × N] → Global Pooling → Linear → z_t
"""

from typing import List

import paddle
import paddle.nn as nn

from layers.causal_conv import CausalConvBlock


class TimbreEncoder(nn.Layer):
    """Convolutional timbre encoder.

    Produces a fixed-dimension speaker embedding from reference mel.

    Args:
        n_mels: Number of mel bins (default 80).
        embed_dim: Timbre embedding dimension (default 256).
        channels: Channel progression (default [32, 64, 128, 256]).
        kernel_sizes: Kernel sizes per block (default [3, 3, 3, 3]).
        strides: Strides per block (default [2, 2, 2, 2]).
    """

    def __init__(
        self,
        n_mels: int = 80,
        embed_dim: int = 256,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        strides: List[int] = None,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 3]
        if strides is None:
            strides = [2, 2, 2, 2]

        assert len(channels) == len(kernel_sizes) == len(strides)

        # Input projection
        self.input_proj = nn.Conv1D(n_mels, channels[0], 1)

        # Causal conv blocks
        blocks = []
        in_ch = channels[0]
        for i, (out_ch, k, s) in enumerate(zip(channels, kernel_sizes, strides)):
            blocks.append(
                CausalConvBlock(in_ch, out_ch, k, stride=s, dilation=1, use_act=True)
            )
            in_ch = out_ch
        self.blocks = nn.LayerList(blocks)

        # Global pooling + projection
        self.pool = nn.AdaptiveAvgPool1D(1)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Extract timbre embedding from reference mel-spectrogram.

        Args:
            mel: (B, n_mels, T) reference mel-spectrogram.

        Returns:
            z_t: (B, embed_dim) global timbre embedding.
        """
        x = self.input_proj(mel)  # (B, C0, T)

        for block in self.blocks:
            x = block(x)  # (B, C_i, T_i)

        # Global average pooling over time
        x = self.pool(x).squeeze(-1)  # (B, C_last)
        z_t = self.proj(x)  # (B, embed_dim)

        return z_t
