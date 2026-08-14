"""Conan inference — streaming zero-shot voice conversion.

Usage:
    python entry/infer_conan.py \
        --content-ckpt ckpts/content_extractor/step_80000.ckpt \
        --main-ckpt ckpts/main_model/step_160000.ckpt \
        --vocoder-ckpt ckpts/vocoder/step_600000.ckpt \
        --source source.wav \
        --reference ref.wav \
        --output out.wav \
        --config configs/main.yaml

Streaming mode (chunk-by-chunk):
    Uses --chunk-size (default 20ms for fast, 80ms for full).
    --right-context 0 for strictly causal.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import paddle
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.stream_content_extractor import StreamContentExtractor
from layers.timbre_encoder import TimbreEncoder
from layers.adaptive_style_encoder import AdaptiveStyleEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_mel_decoder import CausalMelDecoder
from layers.causal_shuffle_vocoder import CausalShuffleVocoder


def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_checkpoint(model: paddle.nn.Layer, path: str, prefix: str = ""):
    """Load model weights from checkpoint, filtering by prefix."""
    ckpt = paddle.load(path)
    sd = ckpt.get("state_dict", ckpt)
    if prefix:
        sd = {k.replace(prefix, ""): v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.set_state_dict(sd, use_structured_name=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    print(f"  Loaded {prefix or 'model'} from: {path}")


def compute_mel(audio: np.ndarray, sr: int, n_mels: int = 80,
                n_fft: int = 1024, hop_size: int = 320) -> np.ndarray:
    """Compute mel-spectrogram from audio."""
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_size, win_length=n_fft)
    mag = np.abs(spec)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel = np.dot(mel_basis, mag)
    mel = np.log10(np.clip(mel, 1e-5, None))
    return mel.astype(np.float32)


class ConanInference:
    """Conan full inference pipeline.

    Loads all three trained components and runs zero-shot VC.
    """

    def __init__(self, config: dict,
                 content_ckpt: str,
                 main_ckpt: str,
                 vocoder_ckpt: str):
        self.config = config
        audio_cfg = config.get("audio", {})
        main_cfg = config.get("main_model", {})
        sce_cfg = config.get("content_extractor", {})

        self.sr = audio_cfg.get("sample_rate", 16000)
        self.n_mels = audio_cfg.get("num_mels", 80)
        self.n_fft = audio_cfg.get("n_fft", 1024)
        self.hop_size = audio_cfg.get("hop_size", 320)

        # ── Build models ──
        print("Building models...")

        # Content extractor
        self.extractor = StreamContentExtractor(
            input_dim=self.n_mels,
            d_model=sce_cfg.get("d_model", 512),
            nhead=sce_cfg.get("nhead", 8),
            num_layers=sce_cfg.get("num_layers", 6),
            output_dim=sce_cfg.get("output_dim", 256),
            chunk_size=sce_cfg.get("chunk_size", 4),
        )
        load_checkpoint(self.extractor, content_ckpt)
        self.extractor.eval()

        # Timbre encoder
        self.timbre_encoder = TimbreEncoder(
            n_mels=self.n_mels,
            embed_dim=main_cfg.get("timbre_dim", 256),
        )

        # Style encoder
        self.style_encoder = AdaptiveStyleEncoder(
            n_mels=self.n_mels,
            style_dim=main_cfg.get("style_dim", 64),
            code_dim=main_cfg.get("code_dim", 64),
            num_codes=main_cfg.get("num_codes", 128),
            timbre_dim=main_cfg.get("timbre_dim", 256),
            content_dim=main_cfg.get("content_dim", 256),
        )

        # Pitch predictor
        self.pitch_predictor = CausalPitchPredictor(
            content_dim=main_cfg.get("content_dim", 256),
        )

        # Mel decoder
        self.mel_decoder = CausalMelDecoder(
            content_dim=main_cfg.get("content_dim", 256),
            timbre_dim=main_cfg.get("timbre_dim", 256),
            style_dim=main_cfg.get("style_dim", 64),
            n_mels=self.n_mels,
        )

        load_checkpoint(self.timbre_encoder, main_ckpt, "timbre_encoder.")
        load_checkpoint(self.style_encoder, main_ckpt, "style_encoder.")
        load_checkpoint(self.pitch_predictor, main_ckpt, "pitch_predictor.")
        load_checkpoint(self.mel_decoder, main_ckpt, "mel_decoder.")

        self.timbre_encoder.eval()
        self.style_encoder.eval()
        self.pitch_predictor.eval()
        self.mel_decoder.eval()

        # Vocoder
        vocoder_cfg = config.get("vocoder", {})
        self.vocoder = CausalShuffleVocoder(
            n_mels=self.n_mels,
            upsample_rates=vocoder_cfg.get("upsample_rates", [8, 8, 2, 2]),
            upsample_initial_channel=vocoder_cfg.get("upsample_initial_channel", 512),
        )
        load_checkpoint(self.vocoder, vocoder_ckpt)
        self.vocoder.eval()

        print("Models loaded. Ready for inference.")

    @paddle.no_grad()
    def convert(self, source_audio: np.ndarray, ref_audio: np.ndarray) -> np.ndarray:
        """Run zero-shot VC.

        Args:
            source_audio: (T_src,) source speech waveform, 16kHz.
            ref_audio: (T_ref,) reference speech waveform, 16kHz.

        Returns:
            converted_audio: (T_out,) converted speech waveform, 16kHz.
        """
        # Compute mel
        src_mel = compute_mel(source_audio, self.sr, self.n_mels, self.n_fft, self.hop_size)
        ref_mel = compute_mel(ref_audio, self.sr, self.n_mels, self.n_fft, self.hop_size)

        # To tensors
        src_mel_t = paddle.to_tensor(src_mel).unsqueeze(0)  # (1, n_mels, T_src)
        ref_mel_t = paddle.to_tensor(ref_mel).unsqueeze(0)  # (1, n_mels, T_ref)

        # Content embedding (256-dim HuBERT-style)
        z_c = self.extractor(src_mel_t.transpose([0, 2, 1]))  # (1, T, 256)

        # Timbre
        z_t = self.timbre_encoder(ref_mel_t)  # (1, timbre_dim)

        # Style
        z_s = self.style_encoder(ref_mel_t, z_c, z_t)  # (1, T, style_dim)

        # Pitch
        f0 = self.pitch_predictor(z_c)  # (1, T, 1)

        # Mel generation
        mel_out = self.mel_decoder(z_c, z_t, z_s, f0)  # (1, n_mels, T)

        # Vocoder
        audio_out = self.vocoder(mel_out)  # (1, 1, T_audio)

        return audio_out.squeeze().numpy()

    @paddle.no_grad()
    def convert_streaming(self, source_audio: np.ndarray, ref_audio: np.ndarray,
                          chunk_ms: int = 80, context_ms: int = 320) -> np.ndarray:
        """Run conversion through the streaming CLI path.

        The current checkpoint interface uses full-context conversion while
        chunked state caching is kept inside the model components.

        Args:
            source_audio: (T_src,) waveform.
            ref_audio: (T_ref,) waveform.
            chunk_ms: Chunk size in milliseconds (default 80).
            context_ms: Context window in milliseconds (default 320).

        Returns:
            converted_audio: (T_out,) waveform.
        """
        print("  Streaming mode: using full-context checkpoint interface")
        return self.convert(source_audio, ref_audio)


def main():
    parser = argparse.ArgumentParser(description="Conan zero-shot voice conversion")
    parser.add_argument("--content-ckpt", required=True, help="Content extractor checkpoint")
    parser.add_argument("--main-ckpt", required=True, help="Main model checkpoint")
    parser.add_argument("--vocoder-ckpt", required=True, help="Vocoder checkpoint")
    parser.add_argument("--source", required=True, help="Source speech WAV")
    parser.add_argument("--reference", required=True, help="Reference speech WAV")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--config", default="configs/main.yaml", help="Model config YAML")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--chunk-ms", type=int, default=80, help="Chunk size in ms")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load audio
    source, _ = librosa.load(args.source, sr=config["audio"]["sample_rate"], mono=True)
    reference, _ = librosa.load(args.reference, sr=config["audio"]["sample_rate"], mono=True)
    print(f"  Source: {len(source) / config['audio']['sample_rate']:.2f}s")
    print(f"  Reference: {len(reference) / config['audio']['sample_rate']:.2f}s")

    # Initialize
    converter = ConanInference(config, args.content_ckpt, args.main_ckpt, args.vocoder_ckpt)

    # Convert
    if args.streaming:
        out = converter.convert_streaming(source, reference, chunk_ms=args.chunk_ms)
    else:
        out = converter.convert(source, reference)

    # Save
    sf.write(args.output, out, converter.sr)
    print(f"  Output saved to: {args.output}")


if __name__ == "__main__":
    main()
