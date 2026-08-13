"""Conan RTF benchmark — measure real-time factor on GPU with/without CINN.

Tests all components with random weights (no training needed).

Reports:
    - Latency per module (ms)
    - RTF (processing_time / audio_duration)
    - With and without CINN (paddle.jit.to_static)
    - Fast setting (20ms chunk, causal) and Full setting (80ms chunk, 2 ctx)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paddle

from layers.stream_content_extractor import StreamContentExtractor
from layers.timbre_encoder import TimbreEncoder
from layers.adaptive_style_encoder import AdaptiveStyleEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_mel_decoder import CausalMelDecoder
from layers.causal_shuffle_vocoder import CausalShuffleVocoder


def rtf_benchmark(label: str, fn, audio_len_s: float,
                   warmup: int = 5, repeat: int = 20,
                   chunk_ms: float = None):
    """Measure RTF.

    Args:
        label: Description.
        fn: Callable to benchmark.
        audio_len_s: Audio duration in seconds.
        warmup: Warmup runs.
        repeat: Timed runs.
        chunk_ms: For per-chunk measurements, the chunk duration in ms.

    Returns:
        (latency_ms, rtf)
    """
    for _ in range(warmup):
        fn()
    paddle.device.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    paddle.device.synchronize()
    elapsed_ms = (time.perf_counter() - start) / repeat * 1000

    chunk_desc = f" (chunk={chunk_ms}ms)" if chunk_ms else ""
    rtf = elapsed_ms / (audio_len_s * 1000)
    realtime = "✅" if rtf < 1.0 else "❌"

    print(f"  {label:55s}  {elapsed_ms:8.2f} ms{chunk_desc:18s}  "
          f"RTF={rtf:.4f}  {realtime}")
    return elapsed_ms, rtf


def main():
    print(f"PaddlePaddle {paddle.__version__}")
    print(f"CUDA: {paddle.is_compiled_with_cuda()}")
    print()

    sr = 16000
    n_mels = 80
    hop_size = 320
    cd = 512  # content_dim
    td = 256  # timbre_dim
    sd = 64   # style_dim

    for audio_len_s in [1.0, 5.0]:
        T = int(audio_len_s * sr / hop_size)  # mel frames

        print(f"{'='*70}")
        print(f"  Audio: {audio_len_s:.0f}s ({T} mel frames)")
        print(f"{'='*70}")

        # ── 1. Stream Content Extractor ──
        print(f"\n── Stream Content Extractor ──")
        for cfg_name, cfg, cinn in [
            ("Fast (20ms, 3L)", dict(num_layers=3, chunk_size=1, right_context=0), False),
            ("Fast (20ms, 3L)", dict(num_layers=3, chunk_size=1, right_context=0), True),
            ("Full (80ms, 6L)", dict(num_layers=6, chunk_size=4, right_context=2), False),
            ("Full (80ms, 6L)", dict(num_layers=6, chunk_size=4, right_context=2), True),
        ]:
            m = StreamContentExtractor(
                input_dim=n_mels, d_model=512, nhead=8,
                num_layers=cfg["num_layers"], output_dim=256,
                chunk_size=cfg["chunk_size"],
                right_context=cfg["right_context"],
            )
            m.eval()
            if cinn:
                m = paddle.jit.to_static(m)
            mel = paddle.randn([1, T, n_mels])
            rtf_benchmark(
                f"SCE {cfg_name}{' [CINN]' if cinn else ''}",
                lambda m=m, x=mel: m(x), audio_len_s,
            )

        # ── 2. Timbre Encoder ──
        print(f"\n── Timbre Encoder ──")
        for cinn in [False, True]:
            m = TimbreEncoder(n_mels=n_mels, embed_dim=td)
            m.eval()
            if cinn:
                m = paddle.jit.to_static(m)
            ref_mel = paddle.randn([1, n_mels, T * 2])
            rtf_benchmark(
                f"Timbre Encoder{' [CINN]' if cinn else ''}",
                lambda m=m, x=ref_mel: m(x), audio_len_s,
            )

        # ── 3. Adaptive Style Encoder ──
        print(f"\n── Adaptive Style Encoder ──")
        for cinn in [False, True]:
            m = AdaptiveStyleEncoder(
                n_mels=n_mels, style_dim=sd, code_dim=64, num_codes=128,
                timbre_dim=td, content_dim=cd,
            )
            m.eval()
            if cinn:
                m = paddle.jit.to_static(m)
            ref = paddle.randn([1, n_mels, T * 2])
            zc = paddle.randn([1, T, cd])
            zt = paddle.randn([1, td])
            rtf_benchmark(
                f"Style Encoder{' [CINN]' if cinn else ''}",
                lambda m=m, a=ref, b=zc, c=zt: m(a, b, c), audio_len_s,
            )

        # ── 4. Causal Pitch Predictor ──
        print(f"\n── Causal Pitch Predictor ──")
        for cinn in [False, True]:
            m = CausalPitchPredictor(content_dim=cd)
            m.eval()
            if cinn:
                m = paddle.jit.to_static(m)
            zc = paddle.randn([1, T, cd])
            rtf_benchmark(
                f"Pitch Predictor{' [CINN]' if cinn else ''}",
                lambda m=m, x=zc: m(x), audio_len_s,
            )

        # ── 5. Causal Mel Decoder ──
        print(f"\n── Causal Mel Decoder ──")
        for cinn in [False, True]:
            m = CausalMelDecoder(
                content_dim=cd, timbre_dim=td, style_dim=sd, n_mels=n_mels,
            )
            m.eval()
            if cinn:
                m = paddle.jit.to_static(m)
            zc = paddle.randn([1, T, cd])
            zt = paddle.randn([1, td])
            zs = paddle.randn([1, T, sd])
            f0 = paddle.randn([1, T, 1])
            rtf_benchmark(
                f"Mel Decoder{' [CINN]' if cinn else ''}",
                lambda m=m, a=zc, b=zt, c=zs, d=f0: m(a, b, c, d),
                audio_len_s,
            )

        # ── 6. Causal Shuffle Vocoder ──
        print(f"\n── Causal Shuffle Vocoder ──")
        for cinn in [False, True]:
            m = CausalShuffleVocoder(
                n_mels=n_mels,
                upsample_rates=[8, 8, 2, 2],
                upsample_initial_channel=512,
            )
            m.eval()
            if cinn:
                m = paddle.jit.to_static(m)
            mel = paddle.randn([1, n_mels, T])
            rtf_benchmark(
                f"Vocoder{' [CINN]' if cinn else ''}",
                lambda m=m, x=mel: m(x), audio_len_s,
            )

        # ── 7. Full Pipeline ──
        print(f"\n── Full Pipeline (all components) ──")
        for cfg_name, sc in [
            ("Fast (20ms, 3L)", dict(num_layers=3, chunk_size=1, right_context=0)),
            ("Full (80ms, 6L)", dict(num_layers=6, chunk_size=4, right_context=2)),
        ]:
            for cinn in [False, True]:
                e = StreamContentExtractor(
                    input_dim=n_mels, d_model=512, nhead=8,
                    num_layers=sc["num_layers"], output_dim=256,
                    chunk_size=sc["chunk_size"],
                    right_context=sc["right_context"],
                )
                t = TimbreEncoder(n_mels=n_mels, embed_dim=td)
                s = AdaptiveStyleEncoder(
                    n_mels=n_mels, style_dim=sd, code_dim=64, num_codes=128,
                    timbre_dim=td, content_dim=cd,
                )
                p = CausalPitchPredictor(content_dim=cd)
                d = CausalMelDecoder(
                    content_dim=cd, timbre_dim=td, style_dim=sd, n_mels=n_mels,
                )
                v = CausalShuffleVocoder(
                    n_mels=n_mels, upsample_rates=[8, 8, 2, 2],
                    upsample_initial_channel=512,
                )
                e.eval(); t.eval(); s.eval(); p.eval(); d.eval(); v.eval()
                if cinn:
                    e = paddle.jit.to_static(e)
                    t = paddle.jit.to_static(t)
                    s = paddle.jit.to_static(s)
                    p = paddle.jit.to_static(p)
                    d = paddle.jit.to_static(d)
                    v = paddle.jit.to_static(v)

                sm = paddle.randn([1, T, n_mels])
                rm = paddle.randn([1, n_mels, T * 2])
                # Project content logits (500 classes) to content_dim (512)
                content_proj = paddle.nn.Linear(500, cd)

                def _run(e=e, t=t, s=s, p=p, d=d, v=v, sm=sm, rm=rm, cp=content_proj):
                    logits = e(sm)          # (1, T, 500)
                    zc = cp(logits)          # (1, T, cd)
                    zt = t(rm)
                    zs = s(rm, zc, zt)
                    f0 = p(zc)
                    mel = d(zc, zt, zs, f0)
                    return v(mel)

                rtf_benchmark(
                    f"Pipeline {cfg_name}{' [CINN]' if cinn else ''}",
                    _run, audio_len_s,
                )

        # ── 8. Per-chunk latency (20ms, causal) ──
        print(f"\n── Per-chunk Overhead (20ms chunks, causal) ──")
        e = StreamContentExtractor(
            input_dim=n_mels, d_model=512, nhead=8,
            num_layers=3, output_dim=256,
            chunk_size=1, right_context=0,
        )
        e.eval()
        mel_proj = e.mel_proj
        emf = e.emformer
        chunk = paddle.randn([1, 1, n_mels])
        mem = paddle.zeros([1, 1, 512])
        summ = paddle.zeros([1, 1, 512])
        lctx = paddle.zeros([1, 1, 512])
        rctx = paddle.zeros([1, 0, 512])

        x = mel_proj(chunk)
        rtf_benchmark(
            "SCE single chunk (20ms)",
            lambda emf=emf, x=x, lctx=lctx, rctx=rctx, mem=mem, summ=summ:
                emf.forward_chunk(x, lctx, rctx, mem, summ),
            0.02, chunk_ms=20,
        )

        # Estimate for full sequence
        num_chunks = T
        chunk_total_ms = num_chunks * 20
        chunk_total_s = chunk_total_ms / 1000
        # Measure full chunked processing
        def _run_all_chunks():
            _mem = paddle.zeros([1, 1, 512])
            _summ = paddle.zeros([1, 1, 512])
            for i in range(num_chunks):
                t_start = i
                t_end = min(i + 2, T)
                _lctx = paddle.randn([1, 1, 512])
                _x = paddle.randn([1, 1, n_mels])
                _rc = paddle.randn([1, 0, 512])
                _x_p = mel_proj(_x)
                _, _mem, _summ = emf.forward_chunk(_x_p, _lctx, _rc, _mem, _summ)

        rtf_benchmark(
            f"All {num_chunks} chunks (20ms × {num_chunks})",
            _run_all_chunks, chunk_total_s, chunk_ms=20,
        )

        print()

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  RTF < 1.0  →  ✅ Real-time capable")
    print("  RTF > 1.0  →  ❌ Slower than real-time")
    print("  Paper: Fast=37ms latency, Full=140ms latency (A100)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
