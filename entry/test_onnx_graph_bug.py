"""Verifying paddle2onnx graph-level bug: MatMul → downstream tensor corruption.

Test insertion of various ops between MatMul and ReduceMean to find workaround.
"""
import sys, os, tempfile, subprocess as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import onnxruntime as ort
import paddle
from paddle.static import InputSpec
import paddle.nn as nn


def test_model(name, model_fn, input_spec):
    """Export model to ONNX and verify numerical correctness."""
    model = model_fn()
    model.eval()
    
    # Generate fixed input
    x = paddle.randn([4, 256])
    eager = model(x).numpy()
    
    sm = paddle.jit.to_static(model, input_spec=[input_spec], full_graph=True)
    
    d = tempfile.mkdtemp(prefix='onnx_')
    paddle.jit.save(sm, os.path.join(d, 'm'))
    r = sp.run(['paddle2onnx', '--model_dir', d,
        '--model_filename', 'm.json', '--params_filename', 'm.pdiparams',
        '--save_file', os.path.join(d, 'out.onnx'),
        '--opset_version', '15', '--enable_onnx_checker', 'True'],
        capture_output=True, text=True, timeout=120)
    
    if r.returncode != 0:
        print(f'  ❌ {name:50s} ONNX export failed')
        return
    
    try:
        sess = ort.InferenceSession(os.path.join(d, 'out.onnx'),
            providers=['CPUExecutionProvider'])
        onnx_out = sess.run(None,
            {sess.get_inputs()[0].name: x.numpy()})[0]
        
        if np.any(np.isnan(onnx_out)):
            print(f'  ❌ {name:50s} NaN in output')
        else:
            diff = np.max(np.abs(eager - onnx_out))
            if diff < 1e-3:
                print(f'  ✅ {name:50s} diff={diff:.3e}')
            else:
                print(f'  ⚠️ {name:50s} diff={diff:.3e} (large)')
    except Exception as e:
        print(f'  ❌ {name:50s} ONNX Runtime error: {e}')


# ── Test cases ──

class MatMulAdd_LN(nn.Layer):
    """Baseline: MatMulAdd + LayerNorm (fails)"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(paddle.matmul(x, self.w) + self.b)

class MatMulAdd_Tanh_LN(nn.Layer):
    """Insert tanh between MatMul and LN"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(paddle.tanh(paddle.matmul(x, self.w) + self.b))

class MatMulAdd_ReLU_LN(nn.Layer):
    """Insert LeakyReLU between MatMul and LN"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(nn.functional.leaky_relu(paddle.matmul(x, self.w) + self.b))

class MatMulAdd_Sigmoid_LN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(nn.functional.sigmoid(paddle.matmul(x, self.w) + self.b))

class MatMulAdd_GELU_LN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(nn.functional.gelu(paddle.matmul(x, self.w) + self.b))

class MatMulAdd_Dropout_LN(nn.Layer):
    """Dropout (eval mode = identity, but graph has dropout op)"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
        self.drop = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(self.drop(paddle.matmul(x, self.w) + self.b))

class Embedding_LN(nn.Layer):
    """Test if Embedding + LN also fails (to isolate the MatMul issue)"""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 128)
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(self.emb(x))

class MatMulAdd_GELU(nn.Layer):
    """MatMulAdd + GELU only (no LN) — should work"""
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([256, 128])
        self.b = self.create_parameter([128])
    def forward(self, x):
        return nn.functional.gelu(paddle.matmul(x, self.w) + self.b)


print("paddle2onnx graph-level bug investigation")
print("=" * 60)
print("Testing MatMul+Add → [activation] → LayerNorm combinations\n")

test_model("MatMulAdd + LN (BASELINE)", MatMulAdd_LN, InputSpec([4, 256], 'float32', 'x'))
test_model("MatMulAdd + Tanh + LN", MatMulAdd_Tanh_LN, InputSpec([4, 256], 'float32', 'x'))
test_model("MatMulAdd + LeakyReLU + LN", MatMulAdd_ReLU_LN, InputSpec([4, 256], 'float32', 'x'))
test_model("MatMulAdd + Sigmoid + LN", MatMulAdd_Sigmoid_LN, InputSpec([4, 256], 'float32', 'x'))
test_model("MatMulAdd + GELU + LN", MatMulAdd_GELU_LN, InputSpec([4, 256], 'float32', 'x'))
test_model("MatMulAdd + Dropout + LN", MatMulAdd_Dropout_LN, InputSpec([4, 256], 'float32', 'x'))
test_model("Embedding + LN (no MatMul)", Embedding_LN, InputSpec([4], 'int64', 'x'))
test_model("MatMulAdd + GELU (no LN)", MatMulAdd_GELU, InputSpec([4, 256], 'float32', 'x'))

print("\n--- Summary ---")
print("If MatMul+LN fails but MatMul+ACT+LN works: bug in paddle2onnx value forwarding")
print("If all fail: general PIR→ONNX bug")
print("If all with LN fail: LayerNorm decomposition bug")
