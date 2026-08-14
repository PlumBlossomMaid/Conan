"""Final ONNX fix verification — all components, numerical + benchmark.

Fixes applied:
1. nn.Linear → LinearViaMatMul (matmul+add + x[:,:] graph barrier)
2. nn.GroupNorm → GroupNormManual (basic ops only)
3. CausalShuffleVocoder — no change needed (already works)
"""
import sys, os, tempfile, subprocess as sp, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import onnxruntime as ort
import paddle
from paddle.static import InputSpec
import paddle.nn as nn

from layers.causal_conv import CausalConvBlock, CausalConv1D
from layers.causal_shuffle_vocoder import CausalShuffleVocoder

N_MELS, CD, TD, SD, T = 80, 512, 256, 64, 50


class LinearViaMatMul(nn.Layer):
    """Linear via matmul+add + x[:,:] graph barrier (avoids linear_v2 PIR op)."""
    def __init__(self, in_f, out_f):
        super().__init__()
        bound = (6.0 / (in_f + out_f)) ** 0.5
        self.w = self.create_parameter([in_f, out_f],
            default_initializer=nn.initializer.Uniform(-bound, bound))
        self.b = self.create_parameter([out_f],
            default_initializer=nn.initializer.Constant(0.0))
    def forward(self, x):
        return (paddle.matmul(x, self.w) + self.b)[:, :]


class GroupNormManual(nn.Layer):
    """GroupNorm with basic ops only (avoids group_norm PIR op)."""
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = self.create_parameter([num_channels],
            default_initializer=nn.initializer.Constant(1.0))
        self.bias = self.create_parameter([num_channels],
            default_initializer=nn.initializer.Constant(0.0))
    def forward(self, x):
        B, C, T = x.shape
        G = self.num_groups
        Cg = C // G
        x = x.reshape([B, G, Cg, T])
        mean = x.mean(axis=[2, 3], keepdim=True)
        var = ((x - mean) ** 2).mean(axis=[2, 3], keepdim=True)
        x = (x - mean) / paddle.sqrt(var + self.eps)
        x = x.reshape([B, C, T])
        return x * self.weight.reshape([1, C, 1]) + self.bias.reshape([1, C, 1])


# ── Components ──

class PitchPredictorONNX(nn.Layer):
    def __init__(self):
        super().__init__()
        hdim = 256
        self.input_proj = LinearViaMatMul(CD, hdim)
        layers = []
        for _ in range(3):
            layers.append(nn.Sequential(
                CausalConv1D(hdim, hdim, 3, dilation=1),
                nn.LeakyReLU(0.2),
                GroupNormManual(1, hdim),
            ))
        self.conv_layers = nn.LayerList(layers)
        self.output = LinearViaMatMul(hdim, 1)
    def forward(self, z_c):
        x = self.input_proj(z_c).transpose([0, 2, 1])
        for layer in self.conv_layers:
            x = layer(x)
        return self.output(x.transpose([0, 2, 1]))


class MelDecoderONNX(nn.Layer):
    def __init__(self):
        super().__init__()
        hdim = 512
        input_dim = CD + TD + SD + 1
        self.input_proj = LinearViaMatMul(input_dim, hdim)
        layers = []
        for i in range(5):
            layers.append(nn.Sequential(
                CausalConv1D(hdim, hdim, 5, dilation=2**i),
                nn.LeakyReLU(0.2),
                GroupNormManual(1, hdim),
            ))
        self.conv_layers = nn.LayerList(layers)
        self.output = nn.Sequential(
            CausalConv1D(hdim, hdim, 3),
            nn.LeakyReLU(0.2),
            CausalConv1D(hdim, N_MELS, 3),
        )
    def forward(self, z_c, z_t, z_s, f0):
        B, Tf = z_c.shape[0], z_c.shape[1]
        z_t_b = z_t.unsqueeze(1).expand([-1, Tf, -1])
        x = paddle.concat([z_c, z_t_b, z_s, f0], axis=-1)
        x = self.input_proj(x).transpose([0, 2, 1])
        for layer in self.conv_layers:
            identity = x
            x = layer(x)
            x = x + identity
        return self.output(x)


# ── Verification ──

def test(name, model_fn, input_fns, specs, warmup=5, repeat=30):
    print(f"\n── {name} ──")
    model = model_fn()
    model.eval()
    inputs = [fn() for fn in input_fns]

    # Eager
    t0 = time.perf_counter()
    for _ in range(repeat):
        model(*inputs)
    paddle.device.synchronize()
    eager_ms = (time.perf_counter() - t0) / repeat * 1000
    eager_out = model(*inputs).numpy()

    # PIR
    sm = paddle.jit.to_static(model, input_spec=specs, full_graph=True)
    sm.eval()
    pir_ins = [paddle.to_tensor(x.numpy()) for x in inputs]
    for _ in range(warmup):
        sm(*pir_ins)
    paddle.device.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        sm(*pir_ins)
    paddle.device.synchronize()
    pir_ms = (time.perf_counter() - t0) / repeat * 1000
    pir_out = sm(*pir_ins).numpy()

    # ONNX
    d = tempfile.mkdtemp(prefix='onnx_')
    paddle.jit.save(sm, os.path.join(d, name))
    r = sp.run(['paddle2onnx', '--model_dir', d,
        '--model_filename', f'{name}.json',
        '--params_filename', f'{name}.pdiparams',
        '--save_file', os.path.join(d, f'{name}.onnx'),
        '--opset_version', '15', '--enable_onnx_checker', 'True'],
        capture_output=True, text=True, timeout=180)

    onnx_path = os.path.join(d, f'{name}.onnx')
    if r.returncode != 0:
        print(f"  ❌ ONNX export: FAILED")
        for line in r.stderr.split('\n')[-3:]:
            if line.strip(): print(f"     {line.strip()[:120]}")
        return

    size_kb = os.path.getsize(onnx_path) / 1024

    try:
        sess = ort.InferenceSession(onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        onnx_ins = {sess.get_inputs()[i].name: inp.numpy() for i, inp in enumerate(inputs)}
        for _ in range(warmup):
            sess.run(None, onnx_ins)
        t0 = time.perf_counter()
        for _ in range(repeat):
            sess.run(None, onnx_ins)
        onnx_ms = (time.perf_counter() - t0) / repeat * 1000
        onnx_out = sess.run(None, onnx_ins)[0]
    except Exception as e:
        print(f"  ❌ ONNX Runtime: {e}")
        onnx_ms = float('nan')
        onnx_out = np.array([float('nan')])

    # Numerics
    diff_ep = float(np.max(np.abs(eager_out - pir_out)))
    diff_po = float(np.max(np.abs(pir_out - onnx_out)))
    diff_eo = float(np.max(np.abs(eager_out - onnx_out)))
    has_nan = np.any(np.isnan(onnx_out))

    if has_nan:
        print(f"  ❌ Numerics: NaN in ONNX output")
    else:
        ok = diff_po < 1e-3 and diff_eo < 1e-3 and diff_ep < 1e-3
        print(f"  {'✅' if ok else '❌'} ONNX={size_kb:7.1f}KB")
        print(f"     E-P={diff_ep:.3e}  P-O={diff_po:.3e}  E-O={diff_eo:.3e}")
        print(f"     Eager={eager_ms:.2f}ms  PIR={pir_ms:.2f}ms  ONNX={onnx_ms:.2f}ms")
        if pir_ms > 0:
            print(f"     Speedup: PIR={eager_ms/pir_ms:.1f}x  ONNX={eager_ms/onnx_ms:.1f}x  (vs eager)")


print(f"Paddle {paddle.__version__}")
print("=" * 70)
print("ONNX fix verification — all components")
print("=" * 70)

# Simple tests first
test("TimbreEncoder",
    lambda: nn.Sequential(
        nn.Conv1D(N_MELS, 32, 1),
        CausalConvBlock(32, 64, 3, stride=2, dilation=1, use_act=True),
        nn.AdaptiveAvgPool1D(1),
        nn.Flatten(),
        LinearViaMatMul(64, TD),
        nn.LayerNorm(TD),
    ),
    [lambda: paddle.randn([1, N_MELS, 100])],
    [InputSpec([1, N_MELS, 100], 'float32', 'mel')])

test("CausalPitchPredictor",
    lambda: PitchPredictorONNX(),
    [lambda: paddle.randn([1, T, CD])],
    [InputSpec([1, T, CD], 'float32', 'z_c')])

test("CausalMelDecoder",
    lambda: MelDecoderONNX(),
    [lambda: paddle.randn([1, T, CD]),
     lambda: paddle.randn([1, TD]),
     lambda: paddle.randn([1, T, SD]),
     lambda: paddle.randn([1, T, 1])],
    [InputSpec([1, T, CD], 'float32', 'z_c'),
     InputSpec([1, TD], 'float32', 'z_t'),
     InputSpec([1, T, SD], 'float32', 'z_s'),
     InputSpec([1, T, 1], 'float32', 'f0')])

# Vocoder — no fix needed, but verify
test("CausalShuffleVocoder",
    lambda: CausalShuffleVocoder(n_mels=N_MELS, upsample_rates=[8,8,2,2], upsample_initial_channel=512),
    [lambda: paddle.randn([1, N_MELS, T])],
    [InputSpec([1, N_MELS, T], 'float32', 'mel')])

print("\n" + "=" * 70)
print("DONE")
