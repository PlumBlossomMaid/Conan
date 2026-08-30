"""Conan Main Model — complete Conan architecture with ocean.Model.

Trains all components together (excluding Content Extractor and Vocoder):

    Adaptive Style Encoder (CVQ) → L_CVQ
    Causal Pitch Predictor       → L_pitch (MSE)
    Causal Mel Decoder           → L_mae + L_ssim + L_GAN (LSGAN)
    Mel Discriminator            → L_GAN (LSGAN)

Training: 160k steps, Adam (beta1=0.9, beta2=0.98), manual G/D optimization.
"""

import os
from typing import Any, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import ocean
from ocean.model import Model

from layers.stream_content_extractor import StreamContentExtractor
from layers.adaptive_style_encoder import AdaptiveStyleEncoder
from layers.timbre_encoder import TimbreEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_mel_decoder import CausalMelDecoder
from layers.causal_shuffle_vocoder import MultiScaleMelDiscriminator
from layers.losses import SSIMMelLoss, GANLoss, FeatureMatchingLoss
from utils.data_utils import build_conan_train_dataloader, build_conan_val_dataloader


def load_content_extractor_from_config(config: dict) -> Optional[StreamContentExtractor]:
    """Build a frozen ``StreamContentExtractor`` from the Stage 1 checkpoint.

    The checkpoint path comes from ``pretrained.content_extractor``, falling
    back to ``data.ckpt_content_extractor``. Stage 1 checkpoints saved by Ocean
    wrap weights in ``state_dict`` under an ``extractor.`` prefix; raw
    ``.pdparams`` state dicts are loaded as-is.

    Args:
        config: Full training configuration dict.

    Returns:
        Frozen extractor in eval mode, or None when no checkpoint is configured.
    """
    ckpt_path = config.get("pretrained", {}).get("content_extractor")
    if not ckpt_path:
        ckpt_path = config.get("data", {}).get("ckpt_content_extractor")
    if not ckpt_path:
        return None
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: content extractor checkpoint not found: {ckpt_path}")
        return None

    sce_cfg = config.get("content_extractor", {})
    audio_cfg = config.get("audio", {})
    extractor = StreamContentExtractor(
        input_dim=audio_cfg.get("num_mels", 80),
        d_model=sce_cfg.get("d_model", 512),
        nhead=sce_cfg.get("nhead", 8),
        num_layers=sce_cfg.get("num_layers", 6),
        output_dim=sce_cfg.get("output_dim", 256),
        chunk_size=sce_cfg.get("chunk_size", 4),
        left_context=sce_cfg.get("left_context", 1),
        right_context=sce_cfg.get("right_context", 2),
    )

    ckpt = paddle.load(ckpt_path)
    if "state_dict" in ckpt:
        extractor.set_state_dict({
            k.replace("extractor.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("extractor.")
        })
    else:
        extractor.set_state_dict(ckpt)

    for p in extractor.parameters():
        p.stop_gradient = True
    extractor.eval()
    print(f"  Loaded content extractor from: {ckpt_path}")
    return extractor


class ConanMainModel(Model):
    """Complete Conan main model for end-to-end training.

    Uses ``automatic_optimization = False`` to manually alternate
    generator and discriminator steps.

    Args:
        config: Full training configuration dict.
        content_extractor: Pretrained ``StreamContentExtractor`` (or None for
                           training from scratch; not recommended).
    """

    def __init__(self, config: dict, content_extractor: Optional[StreamContentExtractor] = None):
        super().__init__()
        self.config = config
        self.automatic_optimization = False

        audio_cfg = config.get("audio", {})
        main_cfg = config.get("main_model", {})
        loss_cfg = config.get("loss", {})

        n_mels = audio_cfg.get("num_mels", 80)
        content_dim = main_cfg.get("content_dim", 256)
        timbre_dim = main_cfg.get("timbre_dim", 256)
        style_dim = main_cfg.get("style_dim", 64)

        # ── Pretrained content extractor (frozen, outputs 256-dim embeddings) ──
        if content_extractor is None:
            content_extractor = load_content_extractor_from_config(config)
        if content_extractor is not None:
            self.content_extractor = content_extractor
            for p in self.content_extractor.parameters():
                p.stop_gradient = True
            self.content_extractor.eval()
        else:
            self.content_extractor = None

        # ── Timbre encoder ──
        self.timbre_encoder = TimbreEncoder(
            n_mels=n_mels,
            embed_dim=timbre_dim,
        )

        # ── Adaptive Style Encoder ──
        self.style_encoder = AdaptiveStyleEncoder(
            n_mels=n_mels,
            style_dim=style_dim,
            code_dim=main_cfg.get("code_dim", 64),
            num_codes=main_cfg.get("num_codes", 128),
            chunk_size=main_cfg.get("style_chunk_size", 4),
            contrastive_weight=loss_cfg.get("contrastive_weight", 0.1),
            timbre_dim=timbre_dim,
            content_dim=content_dim,
        )

        # ── Causal Pitch Predictor ──
        self.pitch_predictor = CausalPitchPredictor(
            content_dim=content_dim,
            hidden_dim=main_cfg.get("pitch_hidden_dim", 256),
            num_layers=main_cfg.get("pitch_num_layers", 3),
        )

        # ── Causal Mel Decoder ──
        self.mel_decoder = CausalMelDecoder(
            content_dim=content_dim,
            timbre_dim=timbre_dim,
            style_dim=style_dim,
            n_mels=n_mels,
            hidden_dim=main_cfg.get("decoder_hidden_dim", 512),
            num_layers=main_cfg.get("decoder_num_layers", 5),
        )

        # ── Mel Discriminator ──
        self.mel_disc = MultiScaleMelDiscriminator()

        # ── Losses ──
        self.ssim_loss = SSIMMelLoss()
        self.gan_loss = GANLoss()
        self.fm_loss = FeatureMatchingLoss()

        # ── Loss weights ──
        self.lambda_mae = loss_cfg.get("mae_weight", 1.0)
        self.lambda_ssim = loss_cfg.get("ssim_weight", 1.0)
        self.lambda_adv = loss_cfg.get("adv_weight", 0.1)
        self.lambda_fm = loss_cfg.get("fm_weight", 2.0)
        self.lambda_pitch = loss_cfg.get("pitch_weight", 1.0)
        self.lambda_vq = loss_cfg.get("vq_weight", 1.0)

    def generator_params(self) -> list:
        """Parameters for generator (everything except the discriminator)."""
        params = list(self.timbre_encoder.parameters())
        params += list(self.style_encoder.parameters())
        params += list(self.pitch_predictor.parameters())
        params += list(self.mel_decoder.parameters())
        return params

    # ── Hub for tensor to manage ──

    def _get_content_embedding(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Get 256-dim content embedding from the frozen content extractor.

        The content extractor distills HuBERT's continuous 256-dim embeddings
        via MSE regression (SVC-verified approach, no clustering needed).

        Args:
            mel: (B, n_mels, T) mel-spectrogram.

        Returns:
            content_emb: (B, T, 256) content embedding.
        """
        if self.content_extractor is not None:
            return self.content_extractor(mel.transpose([0, 2, 1]))
        else:
            # Fallback: simple mel projection (placeholder)
            B, C, T = mel.shape
            proj = paddle.nn.Linear(C, 256)
            return proj(mel.transpose([0, 2, 1]))

    def forward(
        self,
        source_mel: paddle.Tensor,
        ref_mel: paddle.Tensor,
        f0_gt: Optional[paddle.Tensor] = None,
    ) -> dict:
        """Full forward pass.

        Args:
            source_mel: (B, n_mels, T_src) source mel.
            ref_mel: (B, n_mels, T_ref) reference mel.
            f0_gt: (B, T_src, 1) ground truth F0 for loss computation.

        Returns:
            dict with keys: mel_pred, f0_pred, z_t, z_s, stats, losses.
        """
        B = source_mel.shape[0]

        # Content embedding from frozen extractor (256-dim HuBERT embedding)
        z_c = self._get_content_embedding(source_mel)  # (B, T, 256)
        T = z_c.shape[1]

        # Timbre embedding
        z_t = self.timbre_encoder(ref_mel)  # (B, timbre_dim)

        # Style embedding
        z_s = self.style_encoder(ref_mel, z_c, z_t)  # (B, T, style_dim)

        # F0 prediction
        f0_pred = self.pitch_predictor(z_c)  # (B, T, 1)

        # Mel generation
        mel_pred = self.mel_decoder(z_c, z_t, z_s, f0_pred)  # (B, n_mels, T)

        return {
            "mel_pred": mel_pred,
            "f0_pred": f0_pred,
            "z_t": z_t,
            "z_s": z_s,
            "z_c": z_c,
        }

    def _generator_loss(
        self, mel_pred: paddle.Tensor, mel_gt: paddle.Tensor,
        f0_pred: paddle.Tensor, f0_gt: paddle.Tensor,
        valid_mask: Optional[paddle.Tensor] = None,
    ) -> dict:
        """Compute generator losses.

        Losses:
            L_mae: L1 between predicted and GT mel
            L_ssim: 1 - SSIM(pred, GT)
            L_adv: LSGAN generator loss
            L_fm: feature matching loss
            L_pitch: MSE between predicted and GT F0
        """
        B = mel_pred.shape[0]

        # Align lengths
        T_gt = mel_gt.shape[-1]
        T_pred = mel_pred.shape[-1]
        if T_pred > T_gt:
            mel_pred = mel_pred[..., :T_gt]
        elif T_pred < T_gt:
            mel_pred = F.pad(mel_pred, [0, T_gt - T_pred])

        # MAE loss
        if valid_mask is None:
            loss_mae = F.l1_loss(mel_pred, mel_gt) * self.lambda_mae
        else:
            mask = valid_mask.unsqueeze(1)
            loss_mae = (
                (paddle.abs(mel_pred - mel_gt) * mask).sum()
                / (mask.sum() * mel_pred.shape[1]).clip(min=1.0)
                * self.lambda_mae
            )

        # SSIM loss
        loss_ssim = self.ssim_loss(mel_pred, mel_gt, valid_mask) * self.lambda_ssim

        # Adversarial loss (generator)
        # mel_pred: (B, n_mels, T) → (B, 1, n_mels, T)
        fake_feats, fake_logits = self.mel_disc(mel_pred.unsqueeze(1))
        loss_adv = self.gan_loss.generator_loss(fake_logits) * self.lambda_adv

        # Feature matching loss
        real_feats, _ = self.mel_disc(mel_gt.unsqueeze(1))
        loss_fm = self.fm_loss(fake_feats, real_feats) * self.lambda_fm

        # Pitch loss
        if f0_gt is not None:
            T_f0 = f0_gt.shape[1]
            T_pred_f0 = f0_pred.shape[1]
            if T_pred_f0 > T_f0:
                f0_pred = f0_pred[:, :T_f0, :]
            elif T_pred_f0 < T_f0:
                f0_pred = F.pad(f0_pred, [0, 0, 0, T_f0 - T_pred_f0])
            if valid_mask is None:
                loss_pitch = F.mse_loss(f0_pred, f0_gt) * self.lambda_pitch
            else:
                pitch_mask = valid_mask[:, :T_f0].unsqueeze(-1)
                loss_pitch = (
                    ((f0_pred - f0_gt) ** 2) * pitch_mask
                ).sum() / pitch_mask.sum().clip(min=1.0) * self.lambda_pitch
        else:
            loss_pitch = paddle.to_tensor(0.0)

        loss_g = loss_mae + loss_ssim + loss_adv + loss_fm + loss_pitch

        return {
            "loss_g": loss_g,
            "loss_mae": loss_mae,
            "loss_ssim": loss_ssim,
            "loss_adv": loss_adv,
            "loss_fm": loss_fm,
            "loss_pitch": loss_pitch,
        }

    def _discriminator_loss(self, mel_pred: paddle.Tensor, mel_gt: paddle.Tensor) -> paddle.Tensor:
        """Compute discriminator LSGAN loss.

        L_D = 0.5 * mean((D(real) - 1)^2) + 0.5 * mean((D(fake) - 0)^2)
        """
        T_gt = mel_gt.shape[-1]
        T_pred = mel_pred.shape[-1]
        if T_pred > T_gt:
            mel_pred = mel_pred[..., :T_gt]
        elif T_pred < T_gt:
            mel_pred = F.pad(mel_pred, [0, T_gt - T_pred])

        _, logits_real = self.mel_disc(mel_gt.unsqueeze(1))
        _, logits_fake = self.mel_disc(mel_pred.detach().unsqueeze(1))

        return self.gan_loss.discriminator_loss(logits_real, logits_fake)

    def training_step(self, batch: dict, batch_idx: int) -> paddle.Tensor:
        """Training step with gradient accumulation + manual G/D optimization.

        Generator: style encoder + pitch predictor + mel decoder
        Discriminator: MultiScaleMelDiscriminator

        Gradient accumulation:
            Only step optimizers every ``accumulate_grad_batches`` steps.
            Backward runs every step, accumulating gradients across micro-batches.
        """
        acc = self.config.get("training", {}).get("accumulate_grad_batches", 1)
        is_accum_boundary = ((batch_idx + 1) % acc == 0)

        source_mel = batch["source_mel"]
        ref_mel = batch["ref_mel"]
        f0_gt = batch.get("f0")

        opt_g = self._opt_g
        opt_d = self._opt_d

        # ── Shared forward ──
        out = self.forward(source_mel, ref_mel, f0_gt)
        mel_pred = out["mel_pred"]
        mel_gt = batch["source_mel"]

        # ── Generator backward (accumulate grad) ──
        valid_mask = batch.get("source_valid_mask")
        g_losses = self._generator_loss(mel_pred, mel_gt, out["f0_pred"], f0_gt, valid_mask)
        g_losses["loss_g"].backward()

        # ── Discriminator backward (accumulate grad) ──
        loss_d = self._discriminator_loss(mel_pred, mel_gt)
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
            "loss/g_total": g_losses["loss_g"].item(),
            "loss/g_mae": g_losses["loss_mae"].item(),
            "loss/g_ssim": g_losses["loss_ssim"].item(),
            "loss/g_adv": g_losses["loss_adv"].item(),
            "loss/g_fm": g_losses["loss_fm"].item(),
            "loss/g_pitch": g_losses["loss_pitch"].item(),
            "loss/d_total": loss_d.item(),
        })
        self.log("loss/g_total", g_losses["loss_g"].item(), prog_bar=True)
        self.log("loss/d_total", loss_d.item(), prog_bar=True)

        return g_losses["loss_g"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step."""
        with paddle.no_grad():
            source_mel = batch["source_mel"]
            ref_mel = batch["ref_mel"]
            f0_gt = batch.get("f0")

            out = self.forward(source_mel, ref_mel, f0_gt)
            mel_pred = out["mel_pred"]
            mel_gt = source_mel

            g_losses = self._generator_loss(
                mel_pred, mel_gt, out["f0_pred"], f0_gt, batch.get("source_valid_mask")
            )
            loss_d = self._discriminator_loss(mel_pred, mel_gt)

            self.log("val/loss", g_losses["loss_g"].item())

    def train_dataloader(self):
        """Frame-budget training dataloader over the waveform dataset."""
        return build_conan_train_dataloader(self.config)

    def val_dataloader(self):
        """Validation dataloader, capped by ``data.val_max_samples``."""
        return build_conan_val_dataloader(self.config)

    def configure_optimizers(self):
        """Adam optimizers for G and D (manual optimization)."""
        train_cfg = self.config.get("training", {})
        lr = train_cfg.get("learning_rate", 2e-4)

        opt_g = paddle.optimizer.AdamW(
            learning_rate=lr,
            parameters=self.generator_params(),
            beta1=0.9,
            beta2=0.98,
            weight_decay=0.01,
        )
        opt_d = paddle.optimizer.AdamW(
            learning_rate=lr,
            parameters=self.mel_disc.parameters(),
            beta1=0.9,
            beta2=0.98,
            weight_decay=0.01,
        )
        self._opt_g = opt_g
        self._opt_d = opt_d
        return [opt_g, opt_d]
