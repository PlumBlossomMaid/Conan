"""Stream Content Extractor (SCE) — Emformer-based streaming content encoder.

This module distills HuBERT content representations into a streaming
Emformer architecture.

During training, the module can operate in two modes:
1. MSE regression (default): distill continuous 256-dim embeddings.
   This matches the approach verified in SVC4 and is the current default.

2. Cross-entropy classification (paper objective): distill discrete
   content labels. Requires an offline-constructed label codebook via
   `entry/build_label_codebook.py` that clusters HuBERT features offline.

At inference time, it processes audio chunk by chunk with a memory
bank for context continuity, producing 256-dim content embeddings at
50Hz (MSE mode) or discrete labels (CE mode).
"""

from typing import Optional, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from layers.emformer import EmformerEncoder
from layers.label_codebook import PaddleLabelCodebook


class StreamContentExtractor(nn.Layer):
    """Streaming content encoder based on Emformer.

    Produces 256-dim content embeddings (MSE mode) or discrete content
    labels (CE mode) at 20ms intervals from streaming audio, matching
    HuBERT's frame rate (50 Hz).

    MSE mode: wraps an Emformer encoder and an output projection that
    produces 256-dim content embeddings (MSE regression target).

    CE mode: wraps the same Emformer encoder, projects to ``num_labels``
    logits and applies softmax; the training target is the argmax label
    from a k-means codebook over HuBERT features (paper objective).

    Args:
        input_dim: Input feature dimension (e.g., 80 mel bins).
        d_model: Emformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of Emformer layers.
        output_dim: Output content embedding dimension (default 256 = HuBERT dim).
        chunk_size: Frames per chunk (default 4 = 80ms at 50Hz).
        left_context: Number of left context chunks.
        right_context: Number of right context chunks (0 for causal).
        dim_feedforward: FFN dimension.
        dropout: Dropout rate.
        num_labels: Number of discrete content labels (CE mode). When > 0,
                    the head is a linear to ``num_labels`` logits.
        label_codebook: Optional pre-built PaddleLabelCodebook for CE mode.
    """

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        output_dim: int = 256,
        chunk_size: int = 4,
        left_context: int = 1,
        right_context: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_labels: int = 0,
        label_codebook: Optional[PaddleLabelCodebook] = None,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.right_context = right_context
        self.output_dim = output_dim
        self.num_labels = num_labels
        self.label_codebook = label_codebook

        # Input mel projection
        self.mel_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Emformer encoder
        self.emformer = EmformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            left_context=left_context,
            right_context=right_context,
            chunk_size=chunk_size,
        )

        # Content embedding head (regression to 256-dim HuBERT embedding)
        self.content_head = nn.Linear(d_model, output_dim)

        # Classification head (paper objective): d_model → num_labels logits
        if num_labels > 0:
            self.classifier = nn.Linear(d_model, num_labels)
        else:
            self.classifier = None

    def _head(self, x: paddle.Tensor) -> paddle.Tensor:
        """Return the mode-appropriate projection of the encoder output."""
        if self.classifier is not None:
            return self.classifier(x)  # (B, T, num_labels) logits
        return self.content_head(x)  # (B, T, output_dim) embeddings

    def forward(
        self,
        mel: paddle.Tensor,
    ) -> paddle.Tensor:
        """Forward pass (training).

        Args:
            mel: (B, T_mel, n_mels) mel-spectrogram frames.

        Returns:
            content_emb: (B, T_enc, output_dim) 256-dim content embeddings
                         (MSE mode) or (B, T_enc, num_labels) logits (CE mode).
        """
        # Project mel to model dimension
        x = self.mel_proj(mel)  # (B, T, d_model)

        # Emformer encoding
        x = self.emformer(x)  # (B, T, d_model)

        # Mode-appropriate head
        return self._head(x)

    def forward_chunk(
        self,
        mel_chunk: paddle.Tensor,
        left_context: paddle.Tensor,
        right_context: paddle.Tensor,
        memory: paddle.Tensor,
        summary: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Streaming forward for one chunk (inference).

        Args:
            mel_chunk: (B, chunk_size, n_mels) current mel chunk.
            left_context: (B, T_left, d_model) accumulated left context.
            right_context: (B, T_right, d_model) right context frames.
            memory: (B, chunk_size, d_model) memory from previous chunk.
            summary: (B, 1, d_model) summary from previous chunk.

        Returns:
            content_emb: (B, chunk_size, output_dim) content embeddings for this chunk.
            new_memory: (B, chunk_size, d_model) memory for next chunk.
            new_summary: (B, 1, d_model) summary for next chunk.
        """
        x = self.mel_proj(mel_chunk)
        x, new_memory, new_summary = self.emformer.forward_chunk(
            x, left_context, right_context, memory, summary
        )
        return self._head(x), new_memory, new_summary
