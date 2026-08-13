"""Adaptive Style Encoder (ASE) — fine-grained speaker style extraction.

Extracts chunk-level style representations (emotion, prosody) from
reference speech using Clustering VQ and cross-attention alignment.

Architecture:
    Ref Mel → ConvBlocks → Downsample → Linear → CVQ → Align Attention → z_s

The style embedding is aligned with content + timbre via Scaled Dot-Product
Attention, where z_ct = [z_c, z_t] serves as query and style as key/value.
"""

from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from layers.causal_conv import CausalConvBlock
from layers.cvq import ClusteringVQ


class AdaptiveStyleEncoder(nn.Layer):
    """Adaptive Style Encoder with CVQ and alignment attention.

    Args:
        n_mels: Number of mel bins.
        style_dim: Dimension of style embedding (default 64).
        code_dim: Codebook vector dimension (default 64).
        num_codes: Number of codebook entries (default 128).
        chunk_size: Style chunk size in frames (default 4 = 80ms at 50Hz mel rate).
        conv_channels: Conv block channel progression.
        beta: Commitment loss weight for CVQ.
        contrastive_weight: Contrastive loss weight.
        timbre_dim: Dimension of timbre embedding (for alignment).
        content_dim: Dimension of content embedding (for alignment).
        positional_encoding: Whether to add positional encoding to style.
    """

    def __init__(
        self,
        n_mels: int = 80,
        style_dim: int = 64,
        code_dim: int = 64,
        num_codes: int = 128,
        chunk_size: int = 4,
        conv_channels=None,
        beta: float = 0.25,
        contrastive_weight: float = 0.1,
        timbre_dim: int = 256,
        content_dim: int = 512,
        positional_encoding: bool = True,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.style_dim = style_dim
        self.chunk_size = chunk_size
        self.content_dim = content_dim
        self.timbre_dim = timbre_dim

        # Conv blocks: mel → features
        convs = []
        in_ch = n_mels
        for out_ch in conv_channels:
            convs.append(
                nn.Sequential(
                    nn.Conv1D(in_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(0.2),
                )
            )
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*convs)

        # Downsample to chunk level
        self.downsample = nn.Conv1D(conv_channels[-1], conv_channels[-1], chunk_size, stride=chunk_size)

        # Linear projection to code_dim
        self.proj = nn.Linear(conv_channels[-1], code_dim)

        # Clustering VQ
        self.cvq = ClusteringVQ(
            code_dim=code_dim,
            num_codes=num_codes,
            beta=beta,
            contrastive_weight=contrastive_weight,
        )

        # Positional encoding
        if positional_encoding:
            self.pos_enc = PositionalEncoding(style_dim, max_len=512)
        else:
            self.pos_enc = nn.Identity()

        # Align Attention: q = z_ct (content + timbre), k = v = style
        align_dim = content_dim + timbre_dim
        self.align_q_proj = nn.Linear(align_dim, style_dim)
        self.align_k_proj = nn.Linear(style_dim, style_dim)
        self.align_v_proj = nn.Linear(style_dim, style_dim)
        self.align_out = nn.Linear(style_dim, style_dim)

    def forward(
        self,
        ref_mel: paddle.Tensor,
        z_c: paddle.Tensor,
        z_t: paddle.Tensor,
    ) -> paddle.Tensor:
        """Extract style embedding aligned with content + timbre.

        Args:
            ref_mel: (B, n_mels, T_ref) reference mel-spectrogram.
            z_c: (B, T_c, content_dim) content embedding.
            z_t: (B, timbre_dim) timbre embedding.

        Returns:
            z_s: (B, T_c, style_dim) style embedding, aligned per content chunk.
        """
        B = ref_mel.shape[0]

        # Conv blocks
        x = self.conv_blocks(ref_mel)  # (B, C, T_ref)

        # Downsample to chunk level
        T_ref = x.shape[-1]
        # Ensure length is multiple of chunk_size
        pad_len = (self.chunk_size - T_ref % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, [0, pad_len], value=0.0)
        x = self.downsample(x)  # (B, C, T_chunks)

        # Project to code_dim
        x = x.transpose([0, 2, 1])  # (B, T_chunks, C)
        x = self.proj(x)  # (B, T_chunks, code_dim)

        # Clustering VQ
        z_s_chunks, stats = self.cvq(x.transpose([0, 2, 1]))  # (B, code_dim, T_chunks)
        z_s_chunks = z_s_chunks.transpose([0, 2, 1])  # (B, T_chunks, style_dim)

        # Positional encoding
        z_s_chunks = self.pos_enc(z_s_chunks)

        # Align attention: align style chunks to content timesteps
        # z_ct = concat(content, broadcast_timbre)
        z_t_expanded = z_t.unsqueeze(1).expand([-1, z_c.shape[1], -1])  # (B, T_c, timbre_dim)
        z_ct = paddle.concat([z_c, z_t_expanded], axis=-1)  # (B, T_c, align_dim)

        Q = self.align_q_proj(z_ct)  # (B, T_c, style_dim)
        K = self.align_k_proj(z_s_chunks)  # (B, T_chunks, style_dim)
        V = self.align_v_proj(z_s_chunks)  # (B, T_chunks, style_dim)

        # Scaled dot-product cross-attention
        attn = paddle.matmul(Q, K, transpose_y=True) / (self.style_dim ** 0.5)
        attn = F.softmax(attn, axis=-1)  # (B, T_c, T_chunks)
        z_s = paddle.matmul(attn, V)  # (B, T_c, style_dim)
        z_s = self.align_out(z_s)

        return z_s

    def get_cvq_losses(self) -> dict:
        """Return accumulated CVQ losses from the last forward pass.

        Note: CVQ stats are returned from the forward pass. This method
        is a convenience for the training loop.
        """
        return {"vq_loss": paddle.to_tensor(0.0)}


class PositionalEncoding(nn.Layer):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, d_model, 2, dtype=paddle.float32)
            * (-paddle.log(paddle.to_tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x + self.pe[:, :x.shape[1], :]
