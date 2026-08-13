"""Clustering Vector Quantization (CVQ) — adaptive style quantization.

CVQ [Zheng & Vedaldi, CVPR 2023] extends standard VQ with a
dynamic codebook initialization strategy and contrastive loss to
address codebook collapse.

The layer maintains ``num_codes`` code vectors. During training,
each input ``z`` is mapped to the nearest code ``e``, and losses
are computed as:

    L_VQ = ||sg[z] - e||^2 + beta * ||z - sg[e]||^2
    L_contrastive = -log(exp(sim(e, z+)) / sum(exp(sim(e, zi-))))

where ``sg`` is stop-gradient.
"""

from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ClusteringVQ(nn.Layer):
    """Clustering Vector Quantization layer.

    Args:
        code_dim: Dimensionality of each code vector.
        num_codes: Number of codebook entries.
        beta: Commitment loss weight (default 0.25).
        contrastive_weight: Weight for contrastive loss (default 0.1).
        ema_decay: EMA decay for codebook updates (default 0.99); if None, use
                   standard VQ with contrastive loss.
    """

    def __init__(
        self,
        code_dim: int = 64,
        num_codes: int = 128,
        beta: float = 0.25,
        contrastive_weight: float = 0.1,
        ema_decay: Optional[float] = None,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.beta = beta
        self.contrastive_weight = contrastive_weight
        self.ema_decay = ema_decay

        # Codebook: (num_codes, code_dim)
        self.codebook = self.create_parameter(
            shape=[num_codes, code_dim],
            default_initializer=nn.initializer.Uniform(-1.0 / num_codes, 1.0 / num_codes),
        )

        if ema_decay is not None:
            # EMA tracking
            self.register_buffer("_ema_cluster_size", paddle.zeros([num_codes]))
            self.register_buffer("_ema_w", self.codebook.clone())
            # Dummy param to ensure EMA buffers are in state_dict
            self._ema_cluster_size: paddle.Tensor
            self._ema_w: paddle.Tensor

    def forward(
        self, z: paddle.Tensor, return_losses: bool = True
    ) -> Tuple[paddle.Tensor, dict]:
        """Quantize input ``z`` to the nearest codebook entry.

        Args:
            z: (B, C, T) or (B, T, C) input features.
            return_losses: Whether to compute and return loss components.

        Returns:
            z_q: Quantized output (same shape as ``z``).
            stats: Dict with keys ``vq_loss``, ``perplexity``, ``usage``,
                   ``codes`` (indices), and optionally ``contrastive_loss``.
        """
        # Flatten to (B*T, C)
        if z.dim() == 3:
            B, C, T = z.shape
            flat = z.transpose([0, 2, 1]).reshape([-1, C])  # (B*T, C)
        else:
            flat = z
            B, C, T = 1, z.shape[-1], 1

        # Compute distances: (B*T, num_codes)
        dist = (
            paddle.sum(flat ** 2, axis=1, keepdim=True)
            - 2 * paddle.mm(flat, self.codebook.t())
            + paddle.sum(self.codebook.t() ** 2, axis=0, keepdim=True)
        )

        # Nearest code index
        codes = paddle.argmin(dist, axis=1)  # (B*T,)

        # Quantize
        z_q = F.embedding(codes, self.codebook)  # (B*T, C)

        # Restore shape
        if z.dim() == 3:
            z_q = z_q.reshape([B, T, C]).transpose([0, 2, 1])  # (B, C, T)

        # --- Losses ---
        stats = {"codes": codes, "perplexity": 0.0, "usage": 0.0, "vq_loss": 0.0}
        if not return_losses:
            return z_q, stats

        # VQ loss + commitment
        if z.dim() == 3:
            vq_loss = F.mse_loss(z_q, z.detach()) + self.beta * F.mse_loss(z_q.detach(), z)
        else:
            vq_loss = F.mse_loss(z_q, z.detach()) + self.beta * F.mse_loss(z_q.detach(), z)
        stats["vq_loss"] = vq_loss

        # Perplexity (codebook usage)
        if self.training and self.ema_decay is not None:
            # EMA update
            encodings = F.one_hot(codes, self.num_codes).cast(paddle.float32)  # (B*T, num_codes)
            avg_probs = encodings.mean(axis=0)
            perplexity = paddle.exp(-paddle.sum(avg_probs * paddle.log(avg_probs + 1e-10)))
            usage = (encodings.sum(axis=0) > 0).cast(paddle.float32).mean()
            stats["perplexity"] = perplexity
            stats["usage"] = usage
        elif self.training:
            # Standard VQ: compute perplexity from batch
            encodings = F.one_hot(codes, self.num_codes).cast(paddle.float32)
            avg_probs = encodings.mean(axis=0)
            perplexity = paddle.exp(-paddle.sum(avg_probs * paddle.log(avg_probs + 1e-10)))
            usage = (encodings.sum(axis=0) > 0).cast(paddle.float32).mean()
            stats["perplexity"] = perplexity
            stats["usage"] = usage

        # Contrastive loss
        if self.contrastive_weight > 0.0 and self.training:
            contrastive_loss = self._compute_contrastive_loss(flat, codes)
            stats["contrastive_loss"] = contrastive_loss
            stats["vq_loss"] = vq_loss + self.contrastive_weight * contrastive_loss

        return z_q, stats

    def _compute_contrastive_loss(self, z: paddle.Tensor, codes: paddle.Tensor) -> paddle.Tensor:
        """Contrastive loss: pull code closer to its assigned features, push
        away from non-assigned features.

        Args:
            z: (N, code_dim) input features.
            codes: (N,) code indices.

        Returns:
            Scalar contrastive loss.
        """
        # Get the assigned code vectors
        code_vectors = self.codebook  # (num_codes, code_dim)

        # For each code in the batch, find positive pairs (same code) vs negative
        loss = 0.0
        n_unique = 0
        for code_idx in range(self.num_codes):
            mask = (codes == code_idx)
            n_pos = mask.sum()
            if n_pos < 2:
                continue
            n_unique += 1
            pos_features = z[mask]  # (n_pos, code_dim)
            code_vec = code_vectors[code_idx:code_idx + 1]  # (1, code_dim)

            # Positive: similarity between code and its assigned features
            sim_pos = paddle.matmul(pos_features, code_vec.t()).squeeze(-1)  # (n_pos,)

            # Negative: similarity between code and OTHER features
            neg_mask = ~mask
            neg_features = z[neg_mask]  # (n_neg, code_dim)
            if neg_features.shape[0] == 0:
                continue
            # Sample negatives to keep computation manageable
            if neg_features.shape[0] > 128:
                idx = paddle.randperm(neg_features.shape[0])[:128]
                neg_features = neg_features[idx]
            sim_neg = paddle.matmul(neg_features, code_vec.t()).squeeze(-1)  # (n_neg,)

            # InfoNCE: -log(exp(sim_pos) / sum(exp(sim_all)))
            logits = paddle.concat([sim_pos, sim_neg])  # (n_pos + n_neg,)
            labels = paddle.zeros([sim_pos.shape[0]], dtype=paddle.int64)
            loss = loss + F.cross_entropy(
                logits.unsqueeze(0).tile([sim_pos.shape[0], 1]),
                labels.unsqueeze(0).tile([sim_pos.shape[0]]),
            )

        if n_unique == 0:
            return paddle.to_tensor(0.0)

        return loss / n_unique
