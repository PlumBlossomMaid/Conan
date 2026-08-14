"""Conan PIR Inference RTF benchmark (Paddle 3.x).

1. Export each component to static graph via ``paddle.jit.save`` (PIR format)
2. Load via ``paddle.inference`` (C++ PIR runtime)
3. Measure latency + RTF

Usage:
    python entry/bench_inference.py
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import paddle
import paddle.inference as paddle_infer
from paddle.static import InputSpec

from layers.stream_content_extractor import StreamContentExtractor
from layers.timbre_encoder import TimbreEncoder
from layers.adaptive_style_encoder import AdaptiveStyleEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_mel_decoder import CausalMelDecoder
from layers.causal_shuffle_vocoder import CausalShuffleVocoder


def bench_pir(label: str, model_fn, input_gen_fn, audio_len_s: float,
              export_dir: str, warmup: int = 5, repeat: int = 20):
    """Export model to PIR and benchmark with Paddle Inference.

    Args:
        label: Component label.
        model_fn: Callable that returns a fresh model.
        input_gen_fn: Callable that returns list of Tensors (example inputs).
        audio_len_s: Audio length for RTF.
        export_dir: Directory to save exported model.
    """
    name = label.replace(" ", "_").replace("/", "_").replace(",", "")
    print(f"\n  ── {label} ──")

    model = model_fn()
    model.eval()
    example_inputs = input_gen_fn()

    # Build InputSpec from example inputs
    input_specs = [
        InputSpec(x.shape, str(x.dtype).split(".")[-1], f"input_{i}")
        for i, x in enumerate(example_inputs)
    ]
    np_inputs = [x.numpy() for x in example_inputs]

    # ── 1. to_static eager benchmark (baseline) ──
    try:
        static_model = paddle.jit.to_static(model, input_spec=input_specs, full_graph=True)
        static_model.eval()

        # Warmup
        for _ in range(warmup):
            static_model(*[paddle.to_tensor(x) for x in np_inputs])
        paddle.device.synchronize()

        start = time.perf_counter()
        for _ in range(repeat):
            static_model(*[paddle.to_tensor(x) for x in np_inputs])
        paddle.device.synchronize()
        eager_ms = (time.perf_counter() - start) / repeat * 1000
        eager_rtf = eager_ms / (audio_len_s * 1000)
        print(f"    to_static:     {eager_ms:8.2f} ms  "
              f"RTF={eager_rtf:.4f}  {'✅' if eager_rtf < 1.0 else '❌'}")
    except Exception as e:
        print(f"    [SKIP] to_static: {e}")
        eager_ms = None

    # ── 2. Export + Paddle Inference (PIR) ──
    try:
        model_dir = os.path.join(export_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, name)

        paddle.jit.save(static_model, model_path)
        print(f"    Exported → {model_dir}/")
    except Exception as e:
        print(f"    [SKIP] jit.save: {e}")
        return

    try:
        config = paddle_infer.Config(model_dir, name)
        config.enable_use_gpu(256, 0)
        config.switch_ir_optim(True)
        predictor = paddle_infer.create_predictor(config)

        # Set inputs
        for i, (name_i, data) in enumerate(zip(predictor.get_input_names(), np_inputs)):
            h = predictor.get_input_handle(name_i)
            h.reshape(data.shape)
            h.copy_from_cpu(data)

        # Warmup
        for _ in range(warmup):
            predictor.run()
        paddle.device.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(repeat):
            predictor.run()
        paddle.device.synchronize()
        pir_ms = (time.perf_counter() - start) / repeat * 1000
        pir_rtf = pir_ms / (audio_len_s * 1000)

        # Get output shape
        out_name = predictor.get_output_names()[0]
        out_shape = predictor.get_output_handle(out_name).copy_to_cpu().shape

        speedup = f"  ({eager_ms / pir_ms:.1f}x vs to_static)" if eager_ms else ""
        print(f"    PIR Inference: {pir_ms:8.2f} ms  "
              f"RTF={pir_rtf:.4f}  {'✅' if pir_rtf < 1.0 else '❌'}"
              f"{speedup}  out={out_shape}")
    except Exception as e:
        print(f"    [SKIP] PIR Inference: {e}")


def main():
    os.environ["GLOG_minloglevel"] = "2"  # suppress PIR pass logging

    sr = 16000
    n_mels = 80
    hop_size = 320
    cd = 512; td = 256; sd = 64

    export_dir = tempfile.mkdtemp(prefix="conan_pir_")
    print(f"Paddle {paddle.__version__} | CUDA: {paddle.is_compiled_with_cuda()}")
    print(f"Export dir: {export_dir}\n")

    for audio_len_s in [1.0, 5.0]:
        T = int(audio_len_s * sr / hop_size)
        print(f"{'='*60}")
        print(f"  Audio: {audio_len_s:.0f}s  ({T} mel frames)")
        print(f"{'='*60}")

        # ── SCE ──
        for cfg_name, cfg in [
            ("Fast (20ms, 3L)", dict(num_layers=3, chunk_size=1, right_context=0)),
            ("Full (80ms, 6L)", dict(num_layers=6, chunk_size=4, right_context=2)),
        ]:
            bench_pir(f"SCE {cfg_name}",
                lambda c=cfg: StreamContentExtractor(
                    input_dim=n_mels, d_model=512, nhead=8,
                    num_layers=c["num_layers"], output_dim=256,
                    chunk_size=c["chunk_size"], right_context=c["right_context"],
                ),
                lambda: [paddle.randn([1, T, n_mels])],
                audio_len_s, export_dir,
            )

        # ── Timbre ──
        bench_pir("Timbre Encoder",
            lambda: TimbreEncoder(n_mels=n_mels, embed_dim=td),
            lambda: [paddle.randn([1, n_mels, T * 2])],
            audio_len_s, export_dir,
        )

        # ── Style ──
        bench_pir("Style Encoder",
            lambda: AdaptiveStyleEncoder(
                n_mels=n_mels, style_dim=sd, code_dim=64, num_codes=128,
                timbre_dim=td, content_dim=cd,
            ),
            lambda: [paddle.randn([1, n_mels, T * 2]),
                     paddle.randn([1, T, cd]),
                     paddle.randn([1, td])],
            audio_len_s, export_dir,
        )

        # ── Pitch ──
        bench_pir("Pitch Predictor",
            lambda: CausalPitchPredictor(content_dim=cd),
            lambda: [paddle.randn([1, T, cd])],
            audio_len_s, export_dir,
        )

        # ── Mel Decoder ──
        bench_pir("Mel Decoder",
            lambda: CausalMelDecoder(
                content_dim=cd, timbre_dim=td, style_dim=sd, n_mels=n_mels,
            ),
            lambda: [paddle.randn([1, T, cd]),
                     paddle.randn([1, td]),
                     paddle.randn([1, T, sd]),
                     paddle.randn([1, T, 1])],
            audio_len_s, export_dir,
        )

        # ── Vocoder ──
        bench_pir("Vocoder",
            lambda: CausalShuffleVocoder(
                n_mels=n_mels, upsample_rates=[8, 8, 2, 2],
                upsample_initial_channel=512,
            ),
            lambda: [paddle.randn([1, n_mels, T])],
            audio_len_s, export_dir,
        )

        print()

    # Cleanup
    shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
