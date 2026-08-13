"""Full verification: ONNX export with workaround + numerical check + benchmark.

For each component:
1. Apply fixes: Linear→MatMulAdd, GroupNorm→LayerNorm, x[:,:] bypass
2. Export to ONNX
3. Verify: Eager ≈ PIR ≈ ONNX Runtime (numerical)
4. Benchmark latency: Eager vs PIR vs ONNX Runtime
"""
import sys, os, tempfile, subprocess as sp, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import paddle
from paddle.static import InputSpec
import paddle.nn as nn

from layers.causal_conv import CausalConvBlock, CausalConv1D
from layers.causal_shuffle_vocoder import CausalShuffleVocoder

N_MELS, CD, TD, SD, T = 80, 512, 256, 64, 50

# ═══════════════════════════════════════
#  Patched layers
# ═══════════════════════════════════════

class LinearViaMatMul(nn.Layer):
    """nn.Linear replacement using matmul + add + x[:,:] bypass."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w = self.create_parameter([in_f, out_f],
            default_initializer=nn.initializer.XavierUniform())
        self.b = self.create_parameter([out_f],
            default_initializer=nn.initializer.Constant(0.0))
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return h[:, :]  # graph-level identity barrier (fixes paddle2onnx bug)


class GroupNormAsLayerNorm(nn.Layer):
    """GroupNorm(1, C) replacement using LayerNorm (for ONNX compat)."""
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)
    def forward(self, x):
        # x: (B, C, T) → (B, T, C) → LN → (B, T, C) → (B, C, T)
        x = x.transpose([0, 2, 1])
        x = self.ln(x)
        return x.transpose([0, 2, 1])


# ═══════════════════════════════════════
#  Patched components
# ═══════════════════════════════════════

class TimbreEncoderONNX(nn.Layer):
    def __init__(self):
        super().__init__()
        channels = [32, 64, 128, 256]
        self.input_proj = nn.Conv1D(N_MELS, channels[0], 1)
        blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            blocks.append(CausalConvBlock(in_ch, out_ch, 3, stride=2, dilation=1, use_act=True))
            in_ch = out_ch
        self.blocks = nn.LayerList(blocks)
        self.pool = nn.AdaptiveAvgPool1D(1)
        self.proj = nn.Sequential(
            LinearViaMatMul(channels[-1], TD),
            nn.LayerNorm(TD),
        )
    def forward(self, mel):
        x = self.input_proj(mel)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


def make_pitch_predictor_onnx(content_dim=CD):
    class PitchPredictorONNX(nn.Layer):
        def __init__(self):
            super().__init__()
            self.input_proj = LinearViaMatMul(content_dim, 256)
            layers = []
            for _ in range(3):
                layers.append(nn.Sequential(
                    CausalConv1D(256, 256, 3, dilation=1),
                    nn.LeakyReLU(0.2),
                    GroupNormAsLayerNorm(256),
                ))
            self.conv_layers = nn.LayerList(layers)
            self.output = LinearViaMatMul(256, 1)
        def forward(self, z_c):
            x = self.input_proj(z_c)
            x = x.transpose([0, 2, 1])
            for layer in self.conv_layers:
                x = layer(x)
            x = x.transpose([0, 2, 1])
            return self.output(x)
    return PitchPredictorONNX()


def make_mel_decoder_onnx(content_dim=CD, timbre_dim=TD, style_dim=SD):
    class MelDecoderONNX(nn.Layer):
        def __init__(self):
            super().__init__()
            input_dim = content_dim + timbre_dim + style_dim + 1
            self.input_proj = LinearViaMatMul(input_dim, 512)
            layers = []
            for i in range(5):
                layers.append(nn.Sequential(
                    CausalConv1D(512, 512, 5, dilation=2**i),
                    nn.LeakyReLU(0.2),
                    GroupNormAsLayerNorm(512),
                ))
            self.conv_layers = nn.LayerList(layers)
            self.output = nn.Sequential(
                CausalConv1D(512, 512, 3),
                nn.LeakyReLU(0.2),
                CausalConv1D(512, N_MELS, 3),
            )
        def forward(self, z_c, z_t, z_s, f0):
            B, T = z_c.shape[0], z_c.shape[1]
            z_t_b = z_t.unsqueeze(1).expand([-1, T, -1])
            x = paddle.concat([z_c, z_t_b, z_s, f0], axis=-1)
            x = self.input_proj(x)
            x = x.transpose([0, 2, 1])
            for layer in self.conv_layers:
                identity = x
                x = layer(x)
                x = x + identity
            return self.output(x)
    return MelDecoderONNX()


# ═══════════════════════════════════════
#  Test + Verify + Benchmark
# ═══════════════════════════════════════

def test_component(name, model_fn, input_gen, specs, warmup=5, repeat=30):
    print(f"\n── {name} ──")
    model = model_fn()
    model.eval()
    
    # Fixed inputs for verification
    inputs = [fn() for fn in input_gen]
    
    # ── Eager ──
    t0 = time.perf_counter()
    for _ in range(repeat):
        model(*inputs)
    paddle.device.synchronize()
    eager_ms = (time.perf_counter() - t0) / repeat * 1000
    eager_out = model(*inputs)
    
    # ── PIR (to_static) ──  
    sm = paddle.jit.to_static(model, input_spec=specs, full_graph=True)
    sm.eval()
    pir_inputs = [paddle.to_tensor(x.numpy()) for x in inputs]
    
    # Warmup
    for _ in range(warmup):
        sm(*pir_inputs)
    paddle.device.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(repeat):
        sm(*pir_inputs)
    paddle.device.synchronize()
    pir_ms = (time.perf_counter() - t0) / repeat * 1000
    pir_out = sm(*pir_inputs)
    
    # ── ONNX export + Runtime ──
    d = tempfile.mkdtemp(prefix='onnx_')
    paddle.jit.save(sm, os.path.join(d, name))
    
    r = sp.run(['paddle2onnx', '--model_dir', d,
        '--model_filename', f'{name}.json',
        '--params_filename', f'{name}.pdiparams',
        '--save_file', os.path.join(d, f'{name}.onnx'),
        '--opset_version', '15', '--enable_onnx_checker', 'True'],
        capture_output=True, text=True, timeout=180)
    
    onnx_path = os.path.join(d, f'{name}.onnx')
    if r.returncode != 0 or not os.path.exists(onnx_path):
        print(f"  ❌ ONNX export FAILED")
        for line in r.stderr.split('\n')[-3:]:
            if line.strip(): print(f"     {line.strip()[:120]}")
        return
    
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    
    onnx_inputs = {}
    for i, inp in enumerate(inputs):
        onnx_inputs[sess.get_inputs()[i].name] = inp.numpy()
    
    # Warmup
    for _ in range(warmup):
        sess.run(None, onnx_inputs)
    
    t0 = time.perf_counter()
    for _ in range(repeat):
        sess.run(None, onnx_inputs)
    onnx_ms = (time.perf_counter() - t0) / repeat * 1000
    onnx_out = sess.run(None, onnx_inputs)[0]
    
    # ── Numerical verification ──
    diff_ep = np.max(np.abs(eager_out.numpy() - pir_out.numpy()))
    diff_po = np.max(np.abs(pir_out.numpy() - onnx_out))
    diff_eo = np.max(np.abs(eager_out.numpy() - onnx_out))
    
    ok_ep = diff_ep < 1e-5
    ok_po = diff_po < 1e-3
    ok_eo = diff_eo < 1e-3
    
    # ── Results ──
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX: {size_kb:7.1f}KB")
    print(f"  Latency:  Eager={eager_ms:7.2f}ms  PIR={pir_ms:7.2f}ms  ONNX={onnx_ms:7.2f}ms")
    speedup_pir = eager_ms/pir_ms if pir_ms > 0 else 0
    speedup_onnx = eager_ms/onnx_ms if onnx_ms > 0 else 0
    print(f"  Speedup:  PIR={speedup_pir:.2f}x  ONNX={speedup_onnx:.2f}x")
    print(f"  Numerics: E-P={diff_ep:.3e}{'✅' if ok_ep else '❌'}  "
          f"P-O={diff_po:.3e}{'✅' if ok_po else '❌'}  "
          f"E-O={diff_eo:.3e}{'✅' if ok_eo else '❌'}")
    
    return eager_ms, pir_ms, onnx_ms


# ═══ RUN ═══
print(f"Paddle {paddle.__version__}")
print("=" * 70)
print("Full verification: ONNX export + numerical check + benchmark")
print("=" * 70)

# 1. TimbreEncoder
test_component("TimbreEncoder",
    lambda: TimbreEncoderONNX(),
    [lambda: paddle.randn([1, N_MELS, 100])],
    [InputSpec([1, N_MELS, 100], 'float32', 'mel')])

# 2. PitchPredictor
test_component("CausalPitchPredictor",
    lambda: make_pitch_predictor_onnx(),
    [lambda: paddle.randn([1, T, CD])],
    [InputSpec([1, T, CD], 'float32', 'z_c')])

# 3. MelDecoder
test_component("CausalMelDecoder",
    lambda: make_mel_decoder_onnx(),
    [lambda: paddle.randn([1, T, CD]),
     lambda: paddle.randn([1, TD]),
     lambda: paddle.randn([1, T, SD]),
     lambda: paddle.randn([1, T, 1])],
    [InputSpec([1, T, CD], 'float32', 'z_c'),
     InputSpec([1, TD], 'float32', 'z_t'),
     InputSpec([1, T, SD], 'float32', 'z_s'),
     InputSpec([1, T, 1], 'float32', 'f0')])

# 4. Vocoder (already works, verify)
test_component("CausalShuffleVocoder",
    lambda: CausalShuffleVocoder(n_mels=N_MELS, upsample_rates=[8,8,2,2], upsample_initial_channel=512),
    [lambda: paddle.randn([1, N_MELS, T])],
    [InputSpec([1, N_MELS, T], 'float32', 'mel')])

print("\n" + "=" * 70)
print("DONE")
