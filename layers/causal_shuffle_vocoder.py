"""Causal Shuffle Vocoder — fully causal HiFiGAN with pixel-shuffle upsampling.

Replaces transposed convolutions (which cause checkerboard artifacts) with
causal convolutions + pixel shuffle for purely causal upsampling.

Architecture (linear, non-autoregressive):
    Mel → CausalConv → [CausalConv + PixelShuffle + MRF-ResBlocks] × N → CausalConv → Waveform

Each upsampling stage:
    1. CausalConv projects channels: C → r_n * C
    2. PixelShuffle rearranges: (r_n * C, T) → (C, r_n * T)
    3. Parallel MRF-style residual blocks

Based on HiFi-GAN V1 but with all convolutions causal and upsampling via
pixel shuffle instead of transposed conv.
"""

from typing import List, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from layers.causal_conv import CausalConv1D


class CausalMRFResBlock(nn.Layer):
    """Causal multi-receptive-field residual block.

    Same as HiFi-GAN MRF but with causal convolutions.
    """

    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.LayerList()
        self.convs2 = nn.LayerList()

        for d in dilations:
            self.convs1.append(
                CausalConv1D(channels, channels, kernel_size, dilation=d, weight_norm=True)
            )
            self.convs2.append(
                CausalConv1D(channels, channels, kernel_size, dilation=1, weight_norm=True)
            )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        out = 0
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.act(c1(x))
            xt = self.act(c2(xt))
            out = out + xt
        return x + out  # Residual

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            c1.remove_weight_norm()
            c2.remove_weight_norm()


class CausalShuffleVocoder(nn.Layer):
    """Fully causal HiFiGAN vocoder with pixel-shuffle upsampling.

    Args:
        n_mels: Number of mel bins.
        upsample_rates: Ratios per upsampling stage.
        upsample_initial_channel: Initial channel count.
        resblock_kernel_sizes: Kernel sizes for MRF blocks.
        resblock_dilation_sizes: Dilation sequences for MRF blocks.
        hop_size: Audio hop size (for output length calculation).
    """

    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: List[int] = None,
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[int]] = None,
    ):
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 5, 7]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = upsample_rates

        # Initial causal convolution
        self.pre = CausalConv1D(n_mels, upsample_initial_channel, 7, weight_norm=True)

        # Upsampling stages with pixel shuffle
        self.ups = nn.LayerList()
        self.resblocks = nn.LayerList()

        ch = upsample_initial_channel
        for i, u in enumerate(upsample_rates):
            # Causal conv projects to r_n * ch (for pixel shuffle)
            proj_ch = u * ch // 2  # After pixel shuffle: (u*ch/2) → (ch/2) at rate u
            self.ups.append(
                nn.Sequential(
                    CausalConv1D(ch, u * ch // 2, kernel_size=2 * u + 1, weight_norm=True),
                    nn.LeakyReLU(0.2),
                )
            )
            ch = ch // 2  # Channel count after upsampling

            # MRF-style residual blocks
            mrf_blocks = nn.LayerList()
            for k in resblock_kernel_sizes:
                mrf_blocks.append(
                    CausalMRFResBlock(ch, k, resblock_dilation_sizes[i % len(resblock_dilation_sizes)])
                )
            self.resblocks.append(mrf_blocks)

        # Output causal convolution
        self.post = nn.Sequential(
            CausalConv1D(ch, ch, 7, weight_norm=True),
            nn.LeakyReLU(0.2),
            CausalConv1D(ch, 1, 7, weight_norm=True),
        )

    def _pixel_shuffle_1d(self, x: paddle.Tensor, upscale_factor: int) -> paddle.Tensor:
        """1D pixel shuffle: (B, r*C, T) → (B, C, r*T)."""
        B, C_r, T = x.shape
        C = C_r // upscale_factor
        x = x.reshape([B, C, upscale_factor, T])  # (B, C, r, T)
        x = x.transpose([0, 1, 3, 2])              # (B, C, T, r)
        return x.reshape([B, C, T * upscale_factor])  # (B, C, r*T)

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Generate waveform from mel-spectrogram.

        Args:
            mel: (B, n_mels, T_mel) mel-spectrogram.

        Returns:
            audio: (B, 1, T_audio) generated waveform, tanh-limited to [-1, 1].
        """
        x = self.pre(mel)  # (B, init_ch, T_mel)

        for i, (up_stage, mrf_blocks) in enumerate(zip(self.ups, self.resblocks)):
            # Causal conv → pixel shuffle
            x = up_stage(x)  # (B, u_n * ch/2, T)
            # Pixel shuffle (1D): (B, u*ch/2, T) → (B, ch/2, u*T)
            x = self._pixel_shuffle_1d(x, self.upsample_rates[i])

            # MRF residual blocks (averaged)
            mrf_out = 0
            for mrf in mrf_blocks:
                mrf_out = mrf_out + mrf(x)
            x = mrf_out / len(mrf_blocks)

        x = self.post(x)  # (B, 1, T_audio)
        audio = paddle.tanh(x)
        return audio

    def remove_weight_norm(self):
        self.pre.remove_weight_norm()
        for up_stage in self.ups:
            up_stage[0].remove_weight_norm()
        for mrf_blocks in self.resblocks:
            for mrf in mrf_blocks:
                mrf.remove_weight_norm()
        for layer in self.post:
            if hasattr(layer, "remove_weight_norm"):
                layer.remove_weight_norm()


class MelDiscriminator(nn.Layer):
    """Multi-scale discriminator operating on mel-spectrograms.

    Used for adversarial training of the mel decoder (LSGAN-style).

    Architecture:
        Mel → [Conv2D × N] → logits
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.LayerList([
            nn.Conv2D(1, 32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.Conv2D(32, 64, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)),
            nn.Conv2D(64, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)),
            nn.Conv2D(128, 256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.Conv2D(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2D(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ])
        self.act = nn.LeakyReLU(0.2)

    def forward(self, mel: paddle.Tensor) -> Tuple[List[paddle.Tensor], paddle.Tensor]:
        """Forward pass.

        Args:
            mel: (B, 1, n_mels, T) mel-spectrogram.

        Returns:
            feats: List of intermediate feature maps (for feature matching loss).
            logits: (B, 1, H, W) discriminator output.
        """
        feats = []
        x = mel
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = self.act(x)
                feats.append(x)
        return feats, x


class MultiScaleMelDiscriminator(nn.Layer):
    """Multi-scale mel discriminator with different window sizes."""

    def __init__(self, scales: List[Tuple[int, int, int]] = None):
        super().__init__()
        if scales is None:
            scales = [(64, 32, 16), (128, 64, 32), (256, 128, 64)]

        self.discriminators = nn.LayerList([MelDiscriminator() for _ in scales])
        self.poolings = nn.LayerList([
            nn.AvgPool2D(kernel_size=(1, s[0]), stride=(1, s[1]))
            for s in scales
        ])

    def forward(self, mel: paddle.Tensor) -> Tuple[List[List[paddle.Tensor]], List[paddle.Tensor]]:
        """Forward pass through all scales.

        Args:
            mel: (B, 1, n_mels, T).

        Returns:
            all_feats: [[feat_lists for scale 0], ...]
            all_logits: [logits for scale 0, ...]
        """
        all_feats = []
        all_logits = []
        for disc, pool in zip(self.discriminators, self.poolings):
            mel_pooled = pool(mel)
            feats, logits = disc(mel_pooled)
            all_feats.append(feats)
            all_logits.append(logits)
        return all_feats, all_logits
