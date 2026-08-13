"""Emformer encoder — streaming chunk-based Transformer for content extraction.

Based on Emformer (Shi et al., ICASSP 2021), with chunk-wise attention,
left context, right context, and memory bank for streaming inference.

Architecture per layer:
    Q = [C_i, R_i, s_i]  (current chunk, right context, summary)
    K = [M_i, L_i, C_i, R_i]  (memory, left context, current, right)
    V = [M_i, L_i, C_i, R_i]

The summary s_i (mean-pooled C_i) produces the next memory entry m_{i+1}.
"""

from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class EmformerBlock(nn.Layer):
    """Single Emformer block with chunk-wise attention.

    Args:
        d_model: Feature dimension.
        nhead: Number of attention heads.
        dim_feedforward: Hidden dimension of FFN.
        dropout: Dropout rate.
        activation: FFN activation ('relu' or 'gelu').
        left_context: Number of left context chunks.
        right_context: Number of right context chunks.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        left_context: int = 1,
        right_context: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.left_context = left_context
        self.right_context = right_context

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.head_dim = d_model // nhead

        # QKV projections
        self.wq = nn.Linear(d_model, d_model, bias_attr=False)
        self.wk = nn.Linear(d_model, d_model, bias_attr=False)
        self.wv = nn.Linear(d_model, d_model, bias_attr=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_summary = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def _scaled_dot_product_attention(
        self, q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor
    ) -> paddle.Tensor:
        """Scaled dot-product attention.

        Args:
            q: (B, nhead, T_q, head_dim)
            k: (B, nhead, T_kv, head_dim)
            v: (B, nhead, T_kv, head_dim)

        Returns:
            (B, nhead, T_q, head_dim)
        """
        attn = paddle.matmul(q, k, transpose_y=True) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        return paddle.matmul(attn, v)

    def forward(
        self,
        chunk: paddle.Tensor,
        left_context: paddle.Tensor,
        right_context: paddle.Tensor,
        memory: paddle.Tensor,
        summary: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Forward pass for one chunk.

        Args:
            chunk: (B, T_chunk, d_model) — current chunk features.
            left_context: (B, T_left, d_model) — left context.
            right_context: (B, T_right, d_model) — right context.
            memory: (B, T_mem, d_model) — memory bank from previous chunks.
            summary: (B, 1, d_model) — chunk summary (mean pooled).

        Returns:
            chunk_out: (B, T_chunk, d_model) — processed chunk.
            next_summary: (B, 1, d_model) — summary for next chunk's memory.
        """
        B, T_chunk, _ = chunk.shape

        # Construct Q, K, V
        # Q = [chunk, right_context, summary]
        q_input = paddle.concat([chunk, right_context, summary], axis=1)  # (B, T_q, d)

        # K, V = [memory, left_context, chunk, right_context]
        kv_input = paddle.concat([memory, left_context, chunk, right_context], axis=1)  # (B, T_kv, d)

        # Projections
        q = self.wq(q_input)  # (B, T_q, d)
        k = self.wk(kv_input)  # (B, T_kv, d)
        v = self.wv(kv_input)  # (B, T_kv, d)

        # Reshape for multi-head attention
        T_q = q.shape[1]
        T_kv = k.shape[1]
        q = q.reshape([B, T_q, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])  # (B, nH, T_q, hd)
        k = k.reshape([B, T_kv, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])
        v = v.reshape([B, T_kv, self.nhead, self.head_dim]).transpose([0, 2, 1, 3])

        # Attention
        attn_out = self._scaled_dot_product_attention(q, k, v)  # (B, nH, T_q, hd)
        attn_out = attn_out.transpose([0, 2, 1, 3]).reshape([B, T_q, self.d_model])  # (B, T_q, d)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        # Residual + norm (pre-norm style)
        q_input = q_input + attn_out
        q_input = self.norm1(q_input)

        # FFN + residual + norm
        ffn_out = self.ffn(q_input)
        chunk_out = q_input + ffn_out
        chunk_out = self.norm2(chunk_out)

        # Extract output chunk (first T_chunk tokens)
        chunk_processed = chunk_out[:, :T_chunk, :]  # (B, T_chunk, d)

        # Compute next summary from the summary token output
        summary_out = chunk_out[:, T_chunk + right_context.shape[1]: T_chunk + right_context.shape[1] + 1, :]
        next_summary = self.norm_summary(summary_out)

        return chunk_processed, next_summary


class EmformerEncoder(nn.Layer):
    """Stack of Emformer blocks for streaming content encoding.

    Processes input in chunks with left/right context and memory bank.

    Args:
        d_model: Feature dimension.
        nhead: Number of attention heads.
        num_layers: Number of Emformer blocks.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        left_context: Number of left context chunks.
        right_context: Number of right context chunks.
        chunk_size: Frames per chunk (default 4, i.e. 20ms at 50Hz HuBERT frame rate).
        input_projection: Whether to add an input linear projection (if d_input != d_model).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        left_context: int = 1,
        right_context: int = 2,
        chunk_size: int = 4,
        input_projection: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.right_context = right_context

        # Optional input projection
        if input_projection is not None and input_projection != d_model:
            self.input_proj = nn.Linear(input_projection, d_model)
        else:
            self.input_proj = nn.Identity()

        self.layers = nn.LayerList([
            EmformerBlock(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                left_context=left_context,
                right_context=right_context,
            )
            for _ in range(num_layers)
        ])

        # Output projection (logits over HuBERT classes)
        self.output_proj = nn.Linear(d_model, d_model)

    def _split_into_chunks(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, int]:
        """Split input into chunks of chunk_size frames.

        Args:
            x: (B, T, d_model)

        Returns:
            chunks: (B, num_chunks, chunk_size, d_model) — padded if needed.
            T_orig: Original T for padding removal later.
        """
        B, T, D = x.shape
        # Pad to multiple of chunk_size
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, [0, 0, 0, pad_len], value=0.0)
            _, T_padded, _ = x.shape
        else:
            T_padded = T

        num_chunks = T_padded // self.chunk_size
        chunks = x.reshape([B, num_chunks, self.chunk_size, D])
        return chunks, T

    def forward(
        self,
        x: paddle.Tensor,
        num_right_context_frames: Optional[int] = None,
    ) -> paddle.Tensor:
        """Forward pass (training mode, non-streaming).

        Processes the full sequence in chunks, accumulating memory bank.

        Args:
            x: (B, T, d_model) input features (or (B, T, in_dim) if input_projection set).
            num_right_context_frames: Override right context in frames (default: chunk_size * right_context).

        Returns:
            (B, T, d_model) output features — same time resolution as input.
        """
        x = self.input_proj(x)
        B, T, D = x.shape

        chunks, T_orig = self._split_into_chunks(x)
        num_chunks = chunks.shape[1]

        # Right context in chunks
        rc_chunks = num_right_context_frames // self.chunk_size if num_right_context_frames else self.right_context

        # Initialize memory bank (all zeros)
        memory = paddle.zeros([B, self.chunk_size, D], dtype=x.dtype)

        outputs = []
        for i in range(num_chunks):
            cur_chunk = chunks[:, i, :, :]  # (B, chunk_size, D)

            # Left context: previous chunks
            left_start = max(0, i - self.left_context)
            left_frames = []
            for j in range(left_start, i):
                left_frames.append(chunks[:, j, :, :])
            if left_frames:
                left_ctx = paddle.concat(left_frames, axis=1)  # (B, T_left, D)
            else:
                left_ctx = paddle.zeros([B, 0, D], dtype=x.dtype)

            # Right context: future chunks
            right_end = min(num_chunks, i + 1 + rc_chunks)
            right_frames = []
            for j in range(i + 1, right_end):
                right_frames.append(chunks[:, j, :, :])
            if right_frames:
                right_ctx = paddle.concat(right_frames, axis=1)  # (B, T_right, D)
            else:
                right_ctx = paddle.zeros([B, 0, D], dtype=x.dtype)

            # Summary: mean of current chunk
            summary = cur_chunk.mean(axis=1, keepdim=True)  # (B, 1, D)

            # Process through each Emformer layer
            for layer in self.layers:
                cur_chunk, next_summary = layer(
                    cur_chunk, left_ctx, right_ctx, memory, summary
                )
                summary = next_summary

            outputs.append(cur_chunk)

            # Update memory
            memory = cur_chunk  # (B, chunk_size, D) — next iteration's memory

        # Concatenate all chunks
        result = paddle.concat(outputs, axis=1)  # (B, T_padded, D)
        result = result[:, :T_orig, :]  # Remove padding

        return self.output_proj(result)

    def forward_chunk(
        self,
        chunk: paddle.Tensor,
        left_context: paddle.Tensor,
        right_context: paddle.Tensor,
        memory: paddle.Tensor,
        summary: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Forward a single chunk (streaming inference).

        Args:
            chunk: (B, chunk_size, D) current chunk.
            left_context: (B, T_left, D) accumulated left context.
            right_context: (B, T_right, D) right context frames.
            memory: (B, chunk_size, D) memory from previous chunk.
            summary: (B, 1, D) summary from previous chunk.

        Returns:
            chunk_out: (B, chunk_size, D) processed chunk.
            new_memory: (B, chunk_size, D) memory for next chunk.
            new_summary: (B, 1, D) summary for next chunk.
        """
        for layer in self.layers:
            chunk, summary = layer(
                chunk, left_context, right_context, memory, summary
            )
            memory = chunk

        return self.output_proj(chunk), memory, summary
