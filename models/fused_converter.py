"""Fused streaming converter — one PIR model for real-time conversion.

Architecture (Model 2, runs per chunk):
    source_mel + z_t + z_s → SCE → Align Attn → Pitch → Mel Decoder → Vocoder → audio_chunk

Model 1 (Reference Encoder, runs once) is separate:
    ref_mel → TimbreEncoder + StyleEncoder(CVQ) → z_t + z_s

Usage:
    # Stage 1: extract reference embeddings (once)
    z_t, z_s = ref_encoder(ref_mel)

    # Stage 2: streaming conversion (per chunk, fused PIR)
    audio = converter(source_mel_chunk, z_t, z_s)
"""

import paddle
import paddle.nn as nn

from layers.stream_content_extractor import StreamContentExtractor
from layers.timbre_encoder import TimbreEncoder
from layers.adaptive_style_encoder import AdaptiveStyleEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_mel_decoder import CausalMelDecoder
from layers.causal_shuffle_vocoder import CausalShuffleVocoder


class ReferenceEncoder(nn.Layer):
    """Reference encoder — extract timbre + style from reference audio.

    Runs once per reference speaker.

    Args:
        n_mels: Mel bins.
        timbre_dim: Timbre embedding dim.
        style_dim: Style embedding dim.
        content_dim: Content model dim (for style alignment).
        code_dim: CVQ code dimension.
        num_codes: CVQ codebook size.
    """

    def __init__(
        self,
        n_mels: int = 80,
        timbre_dim: int = 256,
        style_dim: int = 64,
        content_dim: int = 512,
        code_dim: int = 64,
        num_codes: int = 128,
    ):
        super().__init__()
        self.timbre_encoder = TimbreEncoder(n_mels=n_mels, embed_dim=timbre_dim)
        self.style_encoder = AdaptiveStyleEncoder(
            n_mels=n_mels, style_dim=style_dim,
            code_dim=code_dim, num_codes=num_codes,
            timbre_dim=timbre_dim, content_dim=content_dim,
        )

    def forward(self, ref_mel: paddle.Tensor) -> tuple:
        """Extract reference embeddings.

        Args:
            ref_mel: (B, n_mels, T_ref) reference mel.

        Returns:
            z_t: (B, timbre_dim) timbre embedding.
            z_s: (B, T_c, style_dim) style embedding (chunk-level).
        """
        z_t = self.timbre_encoder(ref_mel)

        # Style encoder needs z_c too — use a dummy since ref encoder
        # runs before source is seen. z_c is just for positional alignment.
        # During streaming, we re-align per chunk.
        B = ref_mel.shape[0]
        T_c = ref_mel.shape[-1] // 4  # approximate chunk count
        dummy_z_c = paddle.zeros([B, max(1, T_c), self.style_encoder.content_dim])
        z_s = self.style_encoder(ref_mel, dummy_z_c, z_t)
        return z_t, z_s


class FusedStreamingConverter(nn.Layer):
    """Fused streaming converter — full pipeline as one static graph.

    Takes source mel + pre-extracted reference embeddings → audio.

    This is the model that gets exported to PIR for deployment.
    All components are fused into one graph for PIR optimization.

    Args:
        n_mels: Mel bins.
        d_model: Emformer model dimension.
        chunk_size: Emformer chunk size (frames).
        right_context: Emformer right context chunks.
        content_dim: Content feature dimension (matches SCE output_dim).
        timbre_dim: Timbre embedding dimension.
        style_dim: Style embedding dimension.
        pitch_hidden_dim: Pitch predictor hidden dim.
        decoder_hidden_dim: Mel decoder hidden dim.
        upsample_rates: Vocoder upsample rates.
        upsample_initial_channel: Vocoder initial channels.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        chunk_size: int = 4,
        right_context: int = 2,
        left_context: int = 1,
        content_dim: int = 256,
        timbre_dim: int = 256,
        style_dim: int = 64,
        pitch_hidden_dim: int = 256,
        decoder_hidden_dim: int = 512,
        upsample_rates=None,
        upsample_initial_channel: int = 512,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.timbre_dim = timbre_dim
        self.style_dim = style_dim

        # Stream content extractor (SCE) — outputs content_dim continuous embeddings
        self.content_extractor = StreamContentExtractor(
            input_dim=n_mels, d_model=d_model, nhead=nhead,
            num_layers=num_layers, output_dim=content_dim,
            chunk_size=chunk_size, left_context=left_context,
            right_context=right_context,
        )

        # Style alignment (lightweight — just the Align Attention part)
        align_dim = content_dim + timbre_dim
        self.align_q_proj = nn.Linear(align_dim, style_dim)
        self.align_k_proj = nn.Linear(style_dim, style_dim)
        self.align_v_proj = nn.Linear(style_dim, style_dim)
        self.align_out = nn.Linear(style_dim, style_dim)

        # Pitch predictor
        self.pitch_predictor = CausalPitchPredictor(
            content_dim=content_dim, hidden_dim=pitch_hidden_dim,
        )

        # Mel decoder
        self.mel_decoder = CausalMelDecoder(
            content_dim=content_dim, timbre_dim=timbre_dim,
            style_dim=style_dim, n_mels=n_mels,
            hidden_dim=decoder_hidden_dim,
        )

        # Causal shuffle vocoder
        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]
        self.vocoder = CausalShuffleVocoder(
            n_mels=n_mels, upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
        )

    def forward(
        self,
        source_mel: paddle.Tensor,
        z_t: paddle.Tensor,
        z_s: paddle.Tensor,
    ) -> paddle.Tensor:
        """Full streaming conversion — one fused pass.

        Args:
            source_mel: (B, T, n_mels) source mel chunks.
            z_t: (B, timbre_dim) pre-extracted timbre embedding.
            z_s: (B, T_s, style_dim) pre-extracted style embedding.

        Returns:
            audio: (B, 1, T_audio) converted audio waveform.
        """
        B, T, _ = source_mel.shape

        # Content extraction — SCE directly outputs content_dim embeddings
        z_c = self.content_extractor(source_mel)  # (B, T, content_dim)

        # Style alignment (lightweight cross-attention)
        z_t_b = z_t.unsqueeze(1).expand([-1, T, -1])   # (B, T, timbre_dim)
        z_ct = paddle.concat([z_c, z_t_b], axis=-1)     # (B, T, content_dim+timbre)

        Q = self.align_q_proj(z_ct)                     # (B, T, style_dim)
        T_s = z_s.shape[1]
        K = self.align_k_proj(z_s)                      # (B, T_s, style_dim)
        V = self.align_v_proj(z_s)

        attn = paddle.matmul(Q, K, transpose_y=True) / (self.style_dim ** 0.5)
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        z_s_aligned = paddle.matmul(attn, V)            # (B, T, style_dim)
        z_s_aligned = self.align_out(z_s_aligned)

        # Pitch prediction
        f0 = self.pitch_predictor(z_c)                  # (B, T, 1)

        # Mel decoding (mel_decoder broadcasts z_t internally)
        mel = self.mel_decoder(z_c, z_t, z_s_aligned, f0)  # (B, n_mels, T)

        # Vocoder
        audio = self.vocoder(mel)                       # (B, 1, T_audio)

        return audio
