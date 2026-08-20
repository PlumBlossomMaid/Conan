"""F0-guided causal NSF vocoder with pixel-shuffle upsampling."""

from typing import List, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from layers.causal_conv import CausalConv1D


class CausalNSFSource(nn.Layer):
    """Deterministic harmonic source for causal and exportable inference."""

    def __init__(
        self,
        sample_rate: int = 44100,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.voiced_threshold = voiced_threshold
        self.merge = nn.Linear(harmonic_num + 1, 1)
        self.tanh = nn.Tanh()

    def forward(
        self,
        f0: paddle.Tensor,
        samples_per_frame: int,
        rand_ini: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        if f0.ndim == 3:
            if f0.shape[1] != 1:
                raise ValueError(f"Expected F0 shape (B, T) or (B, 1, T), got {f0.shape}")
            f0 = f0.squeeze(1)
        if f0.ndim != 2:
            raise ValueError(f"Expected F0 shape (B, T) or (B, 1, T), got {f0.shape}")

        f0_samples = paddle.repeat_interleave(f0, repeats=samples_per_frame, axis=1)
        harmonics = paddle.arange(
            1, self.harmonic_num + 2, dtype=f0.dtype
        ).reshape([1, 1, -1])
        phase_step = f0_samples.unsqueeze(-1) * harmonics / self.sample_rate
        phase = paddle.cumsum(phase_step, axis=1)
        if rand_ini is not None:
            if rand_ini.ndim == 1:
                rand_ini = rand_ini.unsqueeze(0)
            if rand_ini.shape[-1] != self.harmonic_num + 1:
                raise ValueError(
                    "rand_ini must contain one phase offset per harmonic "
                    f"({self.harmonic_num + 1}), got {rand_ini.shape}"
                )
            phase = phase + rand_ini.unsqueeze(1)
        sine = paddle.sin(phase * (2.0 * 3.141592653589793)) * self.sine_amp
        voiced = (f0_samples > self.voiced_threshold).astype(sine.dtype).unsqueeze(-1)
        sine = sine * voiced
        return self.tanh(self.merge(sine))


class CausalNSFResBlock1(nn.Layer):
    """Causal counterpart of NSF-HiFiGAN's sequential ResBlock1."""

    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.LayerList([
            CausalConv1D(
                channels, channels, kernel_size, dilation=dilation, weight_norm=True
            )
            for dilation in dilations
        ])
        self.convs2 = nn.LayerList([
            CausalConv1D(channels, channels, kernel_size, weight_norm=True)
            for _ in dilations
        ])

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = F.leaky_relu(x, 0.1)
            residual = conv1(residual)
            residual = F.leaky_relu(residual, 0.1)
            x = x + conv2(residual)
        return x

    def remove_weight_norm(self):
        for conv in list(self.convs1) + list(self.convs2):
            conv.remove_weight_norm()


class CausalNSFShuffleVocoder(nn.Layer):
    """44.1 kHz, 50 Hz F0-guided causal NSF vocoder.

    The default product of upsample rates is 882, matching 44100 / 50.
    Inputs use ``(B, n_mels, T)`` mel and ``(B, T)`` or ``(B, 1, T)`` F0.
    """

    def __init__(
        self,
        n_mels: int = 80,
        sample_rate: int = 44100,
        hop_size: int = 882,
        upsample_rates: Optional[List[int]] = None,
        upsample_kernel_sizes: Optional[List[int]] = None,
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: Optional[List[int]] = None,
        resblock_dilation_sizes: Optional[List[List[int]]] = None,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        upsample_rates = upsample_rates or [9, 7, 2, 7]
        upsample_kernel_sizes = upsample_kernel_sizes or [18, 14, 4, 14]
        resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        resblock_dilation_sizes = resblock_dilation_sizes or [[1, 3, 5]] * 3
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must have equal length")
        if sample_rate != hop_size * 50:
            raise ValueError("The default student vocoder requires 50 Hz frames: sample_rate=hop_size*50")

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.upsample_rates = list(upsample_rates)
        self.upsample_kernel_sizes = list(upsample_kernel_sizes)
        self.total_upsample = 1
        for rate in upsample_rates:
            self.total_upsample *= rate
        if self.total_upsample != hop_size:
            raise ValueError(
                f"Upsample product ({self.total_upsample}) must equal hop_size ({hop_size})"
            )

        self.source = CausalNSFSource(
            sample_rate=sample_rate,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            voiced_threshold=voiced_threshold,
        )
        self.pre = CausalConv1D(n_mels, upsample_initial_channel, 7, weight_norm=True)
        self.ups = nn.LayerList()
        self.source_convs = nn.LayerList()
        self.resblocks = nn.LayerList()

        channels = upsample_initial_channel
        cumulative_rate = 1
        for index, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            next_channels = channels // 2
            self.ups.append(nn.Sequential(
                CausalConv1D(
                    channels,
                    rate * next_channels,
                    kernel_size,
                    weight_norm=True,
                ),
                nn.LeakyReLU(0.1),
            ))
            cumulative_rate *= rate
            remaining_rate = self.total_upsample // cumulative_rate
            self.source_convs.append(CausalConv1D(
                1,
                next_channels,
                max(1, 2 * remaining_rate - 1),
                stride=remaining_rate,
                weight_norm=True,
            ))
            self.resblocks.append(nn.LayerList([
                CausalNSFResBlock1(
                    next_channels,
                    kernel,
                    resblock_dilation_sizes[index % len(resblock_dilation_sizes)],
                )
                for kernel in resblock_kernel_sizes
            ]))
            channels = next_channels

        self.post = nn.Sequential(
            CausalConv1D(channels, channels, 7, weight_norm=True),
            nn.LeakyReLU(0.1),
            CausalConv1D(channels, 1, 7, weight_norm=True),
        )

    @staticmethod
    def _pixel_shuffle_1d(x: paddle.Tensor, upscale_factor: int) -> paddle.Tensor:
        batch, channels_times_rate, frames = x.shape
        channels = channels_times_rate // upscale_factor
        x = x.reshape([batch, channels, upscale_factor, frames])
        x = x.transpose([0, 1, 3, 2])
        return x.reshape([batch, channels, frames * upscale_factor])

    def forward(
        self,
        mel: paddle.Tensor,
        f0: paddle.Tensor,
        rand_ini: Optional[paddle.Tensor] = None,
        return_features: bool = False,
    ):
        if mel.ndim != 3:
            raise ValueError(f"Expected mel shape (B, n_mels, T), got {mel.shape}")
        if f0.ndim == 3 and f0.shape[1] == 1:
            f0 = f0.squeeze(1)
        if f0.ndim != 2 or f0.shape[0] != mel.shape[0] or f0.shape[1] != mel.shape[2]:
            raise ValueError(f"F0 shape {f0.shape} does not match mel shape {mel.shape}")

        source = self.source(f0, self.hop_size, rand_ini)
        source = source.transpose([0, 2, 1])
        x = self.pre(mel)
        features = []
        for index, (up, source_conv, blocks) in enumerate(zip(self.ups, self.source_convs, self.resblocks)):
            x = self._pixel_shuffle_1d(up(x), self.upsample_rates[index])
            x = x + source_conv(source)
            block_outputs = [block(x) for block in blocks]
            x = sum(block_outputs) / len(block_outputs)
            features.append(x)
        audio = paddle.tanh(self.post(x))
        if return_features:
            return audio, features
        return audio

    def remove_weight_norm(self):
        self.pre.remove_weight_norm()
        for up in self.ups:
            up[0].remove_weight_norm()
        for source_conv in self.source_convs:
            source_conv.remove_weight_norm()
        for blocks in self.resblocks:
            for block in blocks:
                block.remove_weight_norm()
        for layer in self.post:
            if hasattr(layer, "remove_weight_norm"):
                layer.remove_weight_norm()
