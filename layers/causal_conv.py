"""Causal 1D convolution — core building block for all Conan causal modules.

CausalConv1D pads only on the left side, ensuring each output frame
depends only on current and past input frames (no future leakage).
"""

from typing import Optional

import paddle
import paddle.nn as nn


class CausalConv1D(nn.Layer):
    """1D causal convolution with left-only padding.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride (default 1).
        dilation: Dilation rate (default 1).
        groups: Number of grouped convolutions (default 1).
        bias: Whether to use bias (default True).
        weight_norm: Whether to apply weight normalization (default False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        weight_norm: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Left padding only: pad = (kernel_size - 1) * dilation
        self.pad = (kernel_size - 1) * dilation

        conv = nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=0,  # manual padding
            bias_attr=bias,
        )
        if weight_norm:
            self.conv = nn.utils.weight_norm(conv)
        else:
            self.conv = conv

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward with left-only padding.

        Args:
            x: (B, C, T) input tensor.

        Returns:
            (B, C_out, T_out) output tensor.
        """
        # Pad left side only: (pad_left, pad_right=0)
        x = nn.functional.pad(x, [self.pad, 0], value=0.0)
        return self.conv(x)

    def remove_weight_norm(self):
        if hasattr(self.conv, "remove_weight_norm"):
            self.conv.remove_weight_norm()


class CausalConvBlock(nn.Layer):
    """Causal conv block: CausalConv1D → LeakyReLU (optional).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        stride: Stride.
        dilation: Dilation.
        use_act: Whether to apply LeakyReLU after conv.
        weight_norm: Whether to use weight normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        use_act: bool = True,
        weight_norm: bool = False,
    ):
        super().__init__()
        self.conv = CausalConv1D(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation,
            weight_norm=weight_norm,
        )
        self.act = nn.LeakyReLU(0.2) if use_act else nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.act(self.conv(x))
