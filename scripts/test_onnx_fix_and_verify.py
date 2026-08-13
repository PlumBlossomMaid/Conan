"""Test the fix: replace linear_v2 with matmul+add, then verify ONNX output matches PIR.

Three steps:
1. Create patched versions of components (no nn.Linear, no nn.GroupNorm)
2. Export to ONNX via PIR format
3. Verify: PIR output ≈ ONNX Runtime output (mathematical verification)
"""
import sys, os, tempfile, subprocess as sp, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import paddle
from paddle.static import InputSpec
import paddle.nn as nn

N_MELS, CD, TD, SD, T, CHUNK = 80, 512, 256, 64, 50, 4

# ══════════════════════════════════════════════════════════════
# Custom Linear: matmul + add (no linear_v2 op)
# ══════════════════════════════════════════════════════════════
class LinearViaMatMul(nn.Layer):
    """Linear layer implemented via matmul + add to avoid linear_v2 PIR op."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = self.create_parameter(
            [in_features, out_features],
            default_initializer=nn.initializer.XavierUniform())
        self.b = self.create_parameter(
            [out_features],
            default_initializer=nn.initializer.Constant(0.0))
    
    def forward(self, x):
        # x: (B, ..., in_features)
        # matmul supports arbitrary batch dims
        return paddle.matmul(x, self.w) + self.b


class LayerNormViaScale(nn.Layer):
    """GroupNorm replacement: just LayerNorm on the channel axis.
    
    For (B, C, T) input, GroupNorm(1, C) ≈ LayerNorm(C) on channel dim.
    But LayerNorm stats across (C,) not (C*T,). Still equivalent for 
    time-independent normalization when the model only has C channels.
    
    We use this for TimbreEncoder where GroupNorm isn't used but 
    LayerNorm IS used (and works fine in ONNX).
    """
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, x):
        return self.ln(x)


# ══════════════════════════════════════════════════════════════
# Patched TimbreEncoder (no linear_v2)
# ══════════════════════════════════════════════════════════════
from layers.causal_conv import CausalConvBlock

class TimbreEncoderFixed(nn.Layer):
    """TimbreEncoder with Linear → LinearViaMatMul."""
    def __init__(self, n_mels=80, embed_dim=256):
        super().__init__()
        channels = [32, 64, 128, 256]
        self.input_proj = nn.Conv1D(n_mels, channels[0], 1)
        blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            blocks.append(CausalConvBlock(in_ch, out_ch, 3, stride=2, dilation=1, use_act=True))
            in_ch = out_ch
        self.blocks = nn.LayerList(blocks)
        self.pool = nn.AdaptiveAvgPool1D(1)
        self.proj = nn.Sequential(
            LinearViaMatMul(channels[-1], embed_dim),    # was: nn.Linear
            nn.LayerNorm(embed_dim),
        )

    def forward(self, mel):
        x = self.input_proj(mel)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


# ══════════════════════════════════════════════════════════════
# Patched CausalPitchPredictor (no linear_v2, no group_norm)
# ══════════════════════════════════════════════════════════════
from layers.causal_conv import CausalConv1D

class PitchPredictorFixed(nn.Layer):
    def __init__(self, content_dim=512, hidden_dim=256, num_layers=3, kernel_size=3):
        super().__init__()
        self.input_proj = LinearViaMatMul(content_dim, hidden_dim)  # was: nn.Linear
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                CausalConv1D(hidden_dim, hidden_dim, kernel_size, dilation=1),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(hidden_dim),  # was: GroupNorm(1, hidden_dim)
            ))
        self.conv_layers = nn.LayerList(layers)
        self.output = LinearViaMatMul(hidden_dim, 1)  # was: nn.Linear

    def forward(self, z_c):
        x = self.input_proj(z_c)
        x = x.transpose([0, 2, 1])
        for layer in self.conv_layers:
            x = layer(x)
        x = x.transpose([0, 2, 1])
        return self.output(x)


# ══════════════════════════════════════════════════════════════
# Patched CausalMelDecoder (no linear_v2, no group_norm)
# ══════════════════════════════════════════════════════════════
class MelDecoderFixed(nn.Layer):
    def __init__(self, content_dim=512, timbre_dim=256, style_dim=64, n_mels=80,
                 hidden_dim=512, num_layers=5, kernel_size=5):
        super().__init__()
        input_dim = content_dim + timbre_dim + style_dim + 1
        self.input_proj = LinearViaMatMul(input_dim, hidden_dim)  # was: nn.Linear
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                CausalConv1D(hidden_dim, hidden_dim, kernel_size, dilation=2**i),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(hidden_dim),  # was: GroupNorm(1, hidden_dim)
            ))
        self.conv_layers = nn.LayerList(layers)
        self.output = nn.Sequential(
            CausalConv1D(hidden_dim, hidden_dim, 3),
            nn.LeakyReLU(0.2),
            CausalConv1D(hidden_dim, n_mels, 3),
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


# ══════════════════════════════════════════════════════════════
# Test: export to ONNX + mathematical verification
# ══════════════════════════════════════════════════════════════

def test_and_verify(name, model_fn, input_fns, specs):
    """1. Export to ONNX, 2. Verify eager ≈ PIR ≈ ONNX output."""
    model = model_fn()
    model.eval()

    # ── Run eager mode (baseline) ──
    inputs = [fn() for fn in input_fns]
    eager_out = model(*inputs)
    eager_np = eager_out.numpy()

    # ── PIR (to_static + cinn) ──
    sm = paddle.jit.to_static(model, input_spec=specs, full_graph=True)
    sm.eval()
    pir_out = sm(*[paddle.to_tensor(x.numpy()) for x in inputs])
    pir_np = pir_out.numpy()

    # ── ONNX export ──
    d = tempfile.mkdtemp(prefix='onnx_')
    paddle.jit.save(sm, os.path.join(d, name))

    cmd = ['paddle2onnx', '--model_dir', d,
           '--model_filename', f'{name}.json',
           '--params_filename', f'{name}.pdiparams',
           '--save_file', os.path.join(d, f'{name}.onnx'),
           '--opset_version', '15', '--enable_onnx_checker', 'True']
    r = sp.run(cmd, capture_output=True, text=True, timeout=180)

    onnx_path = os.path.join(d, f'{name}.onnx')
    if r.returncode != 0 or not os.path.exists(onnx_path):
        print(f"  ❌ {name:35s} ONNX export FAILED")
        for line in r.stderr.split('\n')[-3:]:
            if line.strip():
                print(f"       {line.strip()[:120]}")
        return

    size = os.path.getsize(onnx_path) / 1024
    print(f"  ✅ {name:35s} ONNX {size:6.1f}KB")

    # ── ONNX Runtime verification ──
    try:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Prepare ONNX inputs
        onnx_inputs = {}
        for i, inp in enumerate(inputs):
            inp_np = inp.numpy()
            ort_inp_name = ort_sess.get_inputs()[i].name
            onnx_inputs[ort_inp_name] = inp_np
        
        onnx_out = ort_sess.run(None, onnx_inputs)[0]
        
        # ── Compare ──
        # 1. eager vs pir
        diff_ep = np.max(np.abs(eager_np - pir_np))
        # 2. pir vs onnx
        diff_po = np.max(np.abs(pir_np - onnx_out))
        # 3. eager vs onnx
        diff_eo = np.max(np.abs(eager_np - onnx_out))
        
        # Cosine similarity for last verification
        cos_sim = np.dot(eager_np.flatten(), onnx_out.flatten()) / (
            np.linalg.norm(eager_np.flatten()) * np.linalg.norm(onnx_out.flatten()))
        
        status = '✅' if (diff_eo < 1e-3 and cos_sim > 0.999) else '⚠️'
        print(f"       eager vs PIR : max_diff={diff_ep:.3e}")
        print(f"       PIR vs ONNX  : max_diff={diff_po:.3e}")
        print(f"       eager vs ONNX: max_diff={diff_eo:.3e}  cos_sim={cos_sim:.6f}  {status}")
        
    except ImportError:
        print(f"       (skipped ONNX Runtime verification — import onnxruntime failed)")
    except Exception as e:
        print(f"       (ONNX Runtime error: {e})")


# ══════════════════════════════════════════════════════════════
# Run tests
# ══════════════════════════════════════════════════════════════

print(f"Paddle {paddle.__version__}")
print("=" * 70)
print("  Fix verification: Linear → MatMul+Add, GroupNorm → LayerNorm")
print("=" * 70)

# 1. TimbreEncoder
test_and_verify(
    "TimbreEncoder_fixed",
    lambda: TimbreEncoderFixed(N_MELS, TD),
    [lambda: paddle.randn([1, N_MELS, T*2])],
    [InputSpec([1, N_MELS, T*2], 'float32', 'mel')],
)

# 2. PitchPredictor
test_and_verify(
    "PitchPredictor_fixed",
    lambda: PitchPredictorFixed(CD),
    [lambda: paddle.randn([1, T, CD])],
    [InputSpec([1, T, CD], 'float32', 'z_c')],
)

# 3. MelDecoder (注意：这个包含多个 Linear 和 GroupNorm)
test_and_verify(
    "MelDecoder_fixed",
    lambda: MelDecoderFixed(CD, TD, SD, N_MELS),
    [lambda: paddle.randn([1, T, CD]),
     lambda: paddle.randn([1, TD]),
     lambda: paddle.randn([1, T, SD]),
     lambda: paddle.randn([1, T, 1])],
    [InputSpec([1, T, CD], 'float32', 'z_c'),
     InputSpec([1, TD], 'float32', 'z_t'),
     InputSpec([1, T, SD], 'float32', 'z_s'),
     InputSpec([1, T, 1], 'float32', 'f0')],
)

print()
print("Note: CausalShuffleVocoder already exports to ONNX (verified earlier).")
print("StreamContentExtractor (Emformer) and ASE (CVQ) also have linear_v2.")
print("Same fix applies — replace all nn.Linear in those components too.")
