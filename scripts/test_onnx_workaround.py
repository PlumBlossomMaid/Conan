"""Find the MINIMAL workaround for the MatMul→LN tensor corruption bug.

Dropout works but adds overhead. Find cheaper alternatives.
"""
import sys, os, tempfile, subprocess as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import paddle
from paddle.static import InputSpec
import paddle.nn as nn


def test(name, model_fn, spec=InputSpec([4, 256], 'float32', 'x')):
    model = model_fn()
    model.eval()
    x = paddle.randn([4, 256])
    eager = model(x).numpy()
    
    sm = paddle.jit.to_static(model, input_spec=[spec], full_graph=True)
    d = tempfile.mkdtemp(prefix='onnx_')
    paddle.jit.save(sm, os.path.join(d, 'm'))
    r = sp.run(['paddle2onnx', '--model_dir', d,
        '--model_filename', 'm.json', '--params_filename', 'm.pdiparams',
        '--save_file', os.path.join(d, 'out.onnx'),
        '--opset_version', '15', '--enable_onnx_checker', 'True'],
        capture_output=True, text=True, timeout=120)
    
    if r.returncode != 0:
        print(f'  ❌ {name:45s} export failed')
        return False
    
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(d, 'out.onnx'),
        providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    
    if np.any(np.isnan(onnx_out)):
        print(f'  ❌ {name:45s} NaN')
        return False
    
    diff = np.max(np.abs(eager - onnx_out))
    if diff < 1e-3:
        print(f'  ✅ {name:45s} diff={diff:.3e}')
        return True
    elif diff < 1e-1:
        print(f'  ⚠️ {name:45s} diff={diff:.3e}')
        return True
    else:
        print(f'  ❌ {name:45s} diff={diff:.3e} (WRONG)')
        return False


# ═══ BASELINE ═══
class MMA_LN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(paddle.matmul(x, self.w) + self.b)

# ═══ CANDIDATE WORKAROUNDS ═══

class WithScale(nn.Layer):
    """x * 1.0 + 0.0 (cheapest identity)"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        h = h * 1.0 + 0.0
        return self.ln(h)

class WithScaleNoBias(nn.Layer):
    """x * 1.0 only"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        h = h * 1.0
        return self.ln(h)

class WithClone(nn.Layer):
    """paddle.assign (clone)"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return self.ln(h.clone())

class WithIdentity(nn.Layer):
    """nn.Identity wrapper"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.id = nn.Identity()
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return self.ln(self.id(h))

class WithStopGradient(nn.Layer):
    """detach"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return self.ln(h.detach())

class WithSlice(nn.Layer):
    """x[:, :] (identity slice)"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return self.ln(h[:, :])

class WithCast(nn.Layer):
    """cast to same type"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return self.ln(paddle.cast(h, 'float32'))

class WithFullIdentity(nn.Layer):
    """paddle.full_like + add 0"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return self.ln(h + paddle.zeros_like(h))

class WithSqrtNoise(nn.Layer):
    """h + h*0 (identity in math, but more complex graph)"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return self.ln(h + h * 0.0)

class WithPaddleAdd(nn.Layer):
    """paddle.add(x, 0)"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        h = paddle.matmul(x, self.w) + self.b
        return self.ln(paddle.add(h, paddle.zeros_like(h)))


print("Workaround candidates (minimal identity ops between MatMul and LN)")
print("=" * 60, "\n")

baseline = test("BASELINE (no workaround)", MMA_LN)
print()
test("x * 1.0 + 0.0", WithScale)
test("x * 1.0", WithScaleNoBias)
test("x.clone()", WithClone)
test("nn.Identity()", WithIdentity)
test("x.detach()", WithStopGradient)
test("x[:, :]", WithSlice)
test("cast(x, float32)", WithCast)
test("x + zeros_like(x)", WithFullIdentity)
test("x + x*0.0", WithSqrtNoise)
test("add(x, zeros_like(x))", WithPaddleAdd)

print("\n--- Results ---")
print("If *1.0 works → cheapest workaround found")
print("If clone/detach works → cost of extra op")
