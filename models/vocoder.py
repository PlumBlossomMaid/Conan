"""Causal Shuffle Vocoder Model — ocean.Model for vocoder training.

Trains the Causal Shuffle Vocoder with GAN loss + mel loss + feature-map loss.

Training: 600k steps (per paper), with HiFiGAN-style losses.
"""

from typing import Any, List, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import ocean
from ocean.model import Model

from layers.causal_shuffle_vocoder import CausalShuffleVocoder
from layers.losses import GANLoss, FeatureMatchingLoss
from utils.data_utils import build_conan_train_dataloader, build_conan_val_dataloader


class MultiScaleDiscriminatorAudio(nn.Layer):
    """Multi-scale waveform discriminator (HiFiGAN-style).

    Operates at multiple audio scales using average pooling.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.LayerList([
            DiscriminatorAudio(1),
            DiscriminatorAudio(1),
            DiscriminatorAudio(1),
        ])
        self.poolings = nn.LayerList([
            nn.Identity(),
            nn.AvgPool1D(4, stride=2, padding=1),
            nn.AvgPool1D(4, stride=2, padding=1),
        ])
        # Second scale also pools
        self.poolings[1] = nn.AvgPool1D(4, stride=2, padding=1)

    def forward(self, audio: paddle.Tensor) -> tuple:
        """Forward pass.

        Args:
            audio: (B, 1, T).

        Returns:
            all_feats: List of feature lists per scale.
            all_logits: List of logits per scale.
        """
        all_feats, all_logits = [], []
        for disc, pool in zip(self.discriminators, self.poolings):
            x = pool(audio)
            feats, logits = disc(x)
            all_feats.append(feats)
            all_logits.append(logits)
        return all_feats, all_logits


class DiscriminatorAudio(nn.Layer):
    """Single-scale waveform discriminator.

    Architecture follows HiFiGAN period discriminator but applied to
    raw waveform rather than 2D periodic representation.
    """

    def __init__(self, period: int = 1):
        super().__init__()
        self.period = period
        self.convs = nn.LayerList([
            nn.utils.weight_norm(nn.Conv2D(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2D(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2D(128, 256, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2D(256, 512, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2D(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2D(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv2D(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: paddle.Tensor) -> tuple:
        feats = []
        # Reshape for 2D conv: (B, 1, T) → (B, 1, n_periods, period)
        B, C, T = x.shape
        if T % self.period != 0:
            x = F.pad(x, [0, self.period - T % self.period])
        x = x.reshape([B, C, -1, self.period])  # (B, 1, n_periods, period)
        for conv in self.convs:
            x = conv(x)
            x = self.act(x)
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        return feats, x


# Re-assign to avoid circular import
MultiScaleDiscriminatorAudio = MultiScaleDiscriminatorAudio


class VocoderModel(Model):
    """Causal Shuffle Vocoder training model.

    Matches HiFiGAN training setup:
        - GAN loss (LSGAN)
        - Mel loss (L1 between predicted and GT mel)
        - Feature-map loss

    Args:
        config: Training configuration dict.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.automatic_optimization = False

        vocoder_cfg = config.get("vocoder", {})
        audio_cfg = config.get("audio", {})

        self.vocoder = CausalShuffleVocoder(
            n_mels=audio_cfg.get("num_mels", 80),
            upsample_rates=vocoder_cfg.get("upsample_rates", [8, 8, 2, 2]),
            upsample_initial_channel=vocoder_cfg.get("upsample_initial_channel", 512),
        )
        self.mel_disc = MultiScaleDiscriminatorAudio()

        self.gan_loss = GANLoss()
        self.fm_loss = FeatureMatchingLoss()

        self.lambda_mel = config.get("loss", {}).get("mel_weight", 45.0)
        self.lambda_adv = config.get("loss", {}).get("adv_weight", 1.0)
        self.lambda_fm = config.get("loss", {}).get("fm_weight", 10.0)

    def generator_params(self) -> list:
        return list(self.vocoder.parameters())

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Generate waveform from mel.

        Args:
            mel: (B, n_mels, T_mel).

        Returns:
            audio: (B, 1, T_audio).
        """
        return self.vocoder(mel)

    def _mel_loss(self, audio: paddle.Tensor, mel_gt: paddle.Tensor) -> paddle.Tensor:
        """L1 mel loss between predicted audio's mel and GT mel."""
        # Compute mel from generated audio
        # (reuse the same STFT params as GT)
        # This is a simplified version; full HiFiGAN mel loss uses
        # multiple STFT scales.
        audio_mel = self._compute_mel(audio)
        T_gt = mel_gt.shape[-1]
        T_pred = audio_mel.shape[-1]
        if T_pred > T_gt:
            audio_mel = audio_mel[..., :T_gt]
        elif T_pred < T_gt:
            audio_mel = F.pad(audio_mel, [0, T_gt - T_pred])
        return F.l1_loss(audio_mel, mel_gt)

    def _init_mel_basis(self, sr, n_fft, n_mels):
        import numpy as np
        fmax = sr / 2
        def hz2mel(h): return 2595.0 * np.log10(1.0 + h / 700.0)
        def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_pts = np.linspace(hz2mel(0.0), hz2mel(fmax), n_mels + 2)
        hz_pts = mel2hz(mel_pts)
        fft_freqs = np.linspace(0.0, fmax, n_fft // 2 + 1)

        fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
        for i in range(n_mels):
            lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
            for j, f in enumerate(fft_freqs):
                if lo <= f < mid:
                    fb[i, j] = (f - lo) / (mid - lo)
                elif mid <= f <= hi:
                    fb[i, j] = (hi - f) / (hi - mid)
        self.register_buffer("mel_basis", paddle.to_tensor(fb.T, dtype=paddle.float32))

    def _compute_mel(self, audio):
        cfg = self.config["audio"]
        n_fft, hop, win = cfg["n_fft"], cfg["hop_size"], cfg["win_size"]
        n_mels, sr = cfg["num_mels"], cfg["sample_rate"]

        if not hasattr(self, "mel_basis"):
            self._init_mel_basis(sr, n_fft, n_mels)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        spec = paddle.abs(paddle.signal.stft(
            audio, n_fft, hop, win, window=paddle.hann_window(win), center=True))
        mel = paddle.log(paddle.clip(
            paddle.matmul(spec.transpose([0, 2, 1]), self.mel_basis).transpose([0, 2, 1]),
            min=1e-5))
        return mel

    def training_step(self, batch: dict, batch_idx: int) -> paddle.Tensor:
        """Training step with gradient accumulation + manual G/D optimization.

        Generator: vocoder
        Discriminator: MultiScaleDiscriminatorAudio

        Gradient accumulation:
            Only step optimizers every ``accumulate_grad_batches`` steps.
            Backward runs every step, accumulating gradients across micro-batches.
        """
        acc = self.config.get("training", {}).get("accumulate_grad_batches", 1)
        is_accum_boundary = ((batch_idx + 1) % acc == 0)

        mel = batch.get("mel", batch.get("source_mel", batch.get("mel_gt")))
        audio_gt = batch.get("audio", batch.get("source_audio"))

        opt_g = self._opt_g
        opt_d = self._opt_d

        # ── Generator forward ──
        audio_fake = self.vocoder(mel)

        # Align lengths
        T_gt = audio_gt.shape[-1]
        T_fake = audio_fake.shape[-1]
        if T_fake > T_gt:
            audio_fake = audio_fake[..., :T_gt]
        elif T_fake < T_gt:
            audio_fake = F.pad(audio_fake, [0, T_gt - T_fake])

        # ── Generator loss & backward (accumulate grad) ──
        fake_feats, fake_logits = self.mel_disc(audio_fake)
        _, real_logits = self.mel_disc(audio_gt)

        loss_gan = self.gan_loss.generator_loss(fake_logits) * self.lambda_adv
        loss_mel = self._mel_loss(audio_fake, mel) * self.lambda_mel
        loss_fm = self.fm_loss(fake_feats, self._get_real_feats(audio_gt)) * self.lambda_fm

        loss_g = loss_gan + loss_mel + loss_fm
        loss_g.backward()

        # ── Discriminator loss & backward (accumulate grad) ──
        _, real_logits = self.mel_disc(audio_gt)
        _, fake_logits_d = self.mel_disc(audio_fake.detach())
        loss_d = self.gan_loss.discriminator_loss(real_logits, fake_logits_d)
        loss_d.backward()

        # ── Step optimizers at accumulation boundary ──
        if is_accum_boundary:
            paddle.nn.utils.clip_grad_norm_(self.generator_params(), 1.0)
            opt_g.step()
            opt_g.clear_grad()
            opt_d.step()
            opt_d.clear_grad()

        # ── Logging ──
        self.log_dict({
            "loss/g_total": loss_g.item(),
            "loss/g_gan": loss_gan.item(),
            "loss/g_mel": loss_mel.item(),
            "loss/g_fm": loss_fm.item(),
            "loss/d_total": loss_d.item(),
        })
        self.log("loss/g_total", loss_g.item(), prog_bar=True)
        self.log("loss/d_total", loss_d.item(), prog_bar=True)

        return loss_g

    def _get_real_feats(self, audio: paddle.Tensor) -> list:
        """Get discriminator feature maps for real audio."""
        feats, _ = self.mel_disc(audio)
        return feats

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step."""
        with paddle.no_grad():
            mel = batch.get("mel", batch.get("source_mel"))
            audio_gt = batch.get("audio", batch.get("source_audio"))

            audio_fake = self.vocoder(mel)

            T_gt = audio_gt.shape[-1]
            if audio_fake.shape[-1] > T_gt:
                audio_fake = audio_fake[..., :T_gt]
            elif audio_fake.shape[-1] < T_gt:
                audio_fake = F.pad(audio_fake, [0, T_gt - audio_fake.shape[-1]])

            loss_mel = self._mel_loss(audio_fake, mel)
            self.log("val/loss", loss_mel.item())

    def train_dataloader(self):
        """Frame-budget training dataloader over the waveform dataset."""
        return build_conan_train_dataloader(self.config)

    def val_dataloader(self):
        """Validation dataloader, capped by ``data.val_max_samples``."""
        return build_conan_val_dataloader(self.config)

    def configure_optimizers(self):
        """AdamW optimizers for G and D."""
        train_cfg = self.config.get("training", {})
        lr = train_cfg.get("learning_rate", 2e-4)

        opt_g = paddle.optimizer.AdamW(
            learning_rate=lr,
            parameters=self.generator_params(),
            beta1=0.8,
            beta2=0.99,
            weight_decay=0.01,
        )
        opt_d = paddle.optimizer.AdamW(
            learning_rate=lr,
            parameters=self.mel_disc.parameters(),
            beta1=0.8,
            beta2=0.99,
            weight_decay=0.01,
        )
        self._opt_g = opt_g
        self._opt_d = opt_d
        return [opt_g, opt_d]
