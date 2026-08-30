"""Loss functions for Conan training.

Includes:
    - SSIM loss (for mel-spectrogram)
    - Feature matching loss
    - GAN losses (LSGAN)
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SSIMMelLoss(nn.Layer):
    """SSIM loss for mel-spectrogram quality.

    Measures structural similarity between predicted and target
    mel-spectrograms. Values near 1 indicate high similarity.

    Loss = 1 - SSIM(pred, target)
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: paddle.Tensor, target: paddle.Tensor, valid_mask: paddle.Tensor | None = None) -> paddle.Tensor:
        """Compute SSIM loss for valid time frames only."""
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        score = self._ssim(pred, target).mean(axis=1).mean(axis=1)
        if valid_mask is None:
            return 1.0 - score.mean()
        mask = valid_mask
        return 1.0 - (score * mask).sum() / mask.sum().clip(min=1.0)

    def _ssim(self, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
        """Compute SSIM between two images/tensors.

        Tensor dims: (B, C, H, W)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Means
        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.window_size // 2)

        # Variances
        sigma_x = F.avg_pool2d(x ** 2, self.window_size, stride=1, padding=self.window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, self.window_size, stride=1, padding=self.window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1, padding=self.window_size // 2) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        return ssim_n / (ssim_d + 1e-8)


class FeatureMatchingLoss(nn.Layer):
    """Feature matching loss for GAN training.

    L1 distance between discriminator feature maps of real vs fake samples.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, fake_feats: list, real_feats: list
    ) -> paddle.Tensor:
        """Compute feature matching loss.

        Args:
            fake_feats: List of feature maps from fake samples.
            real_feats: List of feature maps from real samples.

        Returns:
            Scalar loss.
        """
        loss = 0.0
        n_layers = 0
        for fake_scale, real_scale in zip(fake_feats, real_feats):
            for fake_f, real_f in zip(fake_scale, real_scale):
                loss = loss + F.l1_loss(fake_f, real_f.detach())
                n_layers += 1
        return loss / max(n_layers, 1)


class GANLoss(nn.Layer):
    """LSGAN loss for discriminator and generator."""

    def __init__(self):
        super().__init__()

    def discriminator_loss(
        self, real_logits: list, fake_logits: list
    ) -> paddle.Tensor:
        """LSGAN discriminator loss.

        L_D = 0.5 * mean((D(real) - 1)^2) + 0.5 * mean((D(fake) - 0)^2)
        """
        loss_real = sum(F.mse_loss(l, paddle.ones_like(l)) for l in real_logits)
        loss_fake = sum(F.mse_loss(l, paddle.zeros_like(l)) for l in fake_logits)
        n = len(real_logits) + len(fake_logits)
        return (loss_real + loss_fake) / max(n, 1)

    def generator_loss(self, fake_logits: list) -> paddle.Tensor:
        """LSGAN generator loss.

        L_G = mean((D(fake) - 1)^2)
        """
        return sum(F.mse_loss(l, paddle.ones_like(l)) for l in fake_logits) / max(len(fake_logits), 1)
