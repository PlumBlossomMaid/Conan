"""Benchmark fused streaming converter with full PIR + CINN compilation.

Usage:
    nohup python entry/bench_fused_pir.py > bench_fused.log 2>&1 &
    tail -f bench_fused.log
"""

import sys, os, time, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import paddle
from paddle.static import InputSpec
import paddle.inference as infer
from models.fused_converter import FusedStreamingConverter


def benchmark(T: int, desc: str, full_graph: bool = True):
    sr, n_mels, hop = 16000, 80, 320
    cd, td, sd = 512, 256, 64
    audio_len_s = T * hop / sr

    sep = '=' * 60
    print(sep)
    print(f'  {desc}  ({T} frames = {audio_len_s:.1f}s)')
    print(sep)

    model = FusedStreamingConverter(
        n_mels=n_mels, num_layers=6, chunk_size=4, right_context=2,
        content_dim=cd, timbre_dim=td, style_dim=sd,
    )
    model.eval()

    src_mel = paddle.randn([1, T, n_mels])
    z_t = paddle.randn([1, td])
    z_s = paddle.randn([1, T // 4 + 1, sd])

    specs = [
        InputSpec([1, T, n_mels], 'float32', 'source_mel'),
        InputSpec([1, td], 'float32', 'z_t'),
        InputSpec([1, T // 4 + 1, sd], 'float32', 'z_s'),
    ]

    # Eager baseline
    for _ in range(3):
        model(src_mel, z_t, z_s)
    paddle.device.synchronize()
    n = time.perf_counter()
    for _ in range(10):
        model(src_mel, z_t, z_s)
    paddle.device.synchronize()
    e_ms = (time.perf_counter() - n) / 10 * 1000
    print(f'  Eager:       {e_ms:8.2f} ms  RTF={e_ms/1000/audio_len_s:.4f}')

    # Compile + export + PIR
    t_start = time.time()
    sm = paddle.jit.to_static(model, input_spec=specs, full_graph=full_graph)
    sm.eval()
    t_compile = time.time() - t_start
    print(f'  CINN compile: {t_compile:.0f}s')

    # to_static benchmark
    for _ in range(3):
        sm(src_mel, z_t, z_s)
    paddle.device.synchronize()
    n = time.perf_counter()
    for _ in range(10):
        sm(src_mel, z_t, z_s)
    paddle.device.synchronize()
    s_ms = (time.perf_counter() - n) / 10 * 1000
    print(f'  to_static:   {s_ms:8.2f} ms  RTF={s_ms/1000/audio_len_s:.4f}')

    # PIR export
    export_dir = tempfile.mkdtemp(prefix='fused_pir_')
    t0 = time.time()
    paddle.jit.save(sm, os.path.join(export_dir, 'converter'))
    t_export = time.time() - t0

    # PIR load + optimize
    t0 = time.time()
    cfg = infer.Config(export_dir, 'converter')
    cfg.enable_use_gpu(256, 0)
    pr = infer.create_predictor(cfg)
    t_load = time.time() - t0
    print(f'  PIR save: {t_export:.0f}s  load+optimize: {t_load:.0f}s')

    # PIR inference
    for ni, arr in zip(pr.get_input_names(), [src_mel.numpy(), z_t.numpy(), z_s.numpy()]):
        h = pr.get_input_handle(ni)
        h.reshape(arr.shape)
        h.copy_from_cpu(arr)

    for _ in range(3):
        pr.run()
    paddle.device.synchronize()
    n = time.perf_counter()
    for _ in range(10):
        pr.run()
    paddle.device.synchronize()
    p_ms = (time.perf_counter() - n) / 10 * 1000
    out = pr.get_output_handle(pr.get_output_names()[0]).copy_to_cpu()
    print(f'  PIR:         {p_ms:8.2f} ms  RTF={p_ms/1000/audio_len_s:.4f}')
    print(f'  Speedup: {e_ms/p_ms:.1f}x vs eager, {s_ms/p_ms:.1f}x vs static')
    print(f'  Output: {out.shape}')

    # Model size
    total = sum(os.path.getsize(os.path.join(export_dir, f)) for f in os.listdir(export_dir))
    print(f'  Export: {total/1024/1024:.1f} MB  ({os.listdir(export_dir)})')

    shutil.rmtree(export_dir, ignore_errors=True)
    print()


if __name__ == '__main__':
    # Fast setting: 20ms chunk = 1 frame
    # Full setting: 80ms chunk = 4 frames, 6 layers
    # Test at 1s chunks for per-chunk latency
    benchmark(T=50, desc='Full Setting (80ms, 6L)  —  1s chunk')
    benchmark(T=250, desc='Full Setting (80ms, 6L)  —  5s chunk')
