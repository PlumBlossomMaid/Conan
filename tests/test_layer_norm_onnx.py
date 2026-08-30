"""Test if LayerNorm + MatMulAdd works in ONNX."""
import sys, os, tempfile, subprocess as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import onnx
import onnxruntime as ort
import paddle
from paddle.static import InputSpec
import paddle.nn as nn


class LinearViaMatMul(nn.Layer):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w = self.create_parameter([in_f, out_f],
            default_initializer=nn.initializer.XavierUniform())
        self.b = self.create_parameter([out_f],
            default_initializer=nn.initializer.Constant(0.0))
    def forward(self, x):
        return paddle.matmul(x, self.w) + self.b


class SmallModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.l1 = LinearViaMatMul(256, 128)
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(self.l1(x))


# ── Test 1: MatMulAdd + LayerNorm ──
print("Test 1: LinearViaMatMul(256→128) + LayerNorm(128)")
model = SmallModel()
model.eval()
x = paddle.randn([4, 256])
eager = model(x).numpy()

sm = paddle.jit.to_static(model,
    input_spec=[InputSpec([4, 256], 'float32', 'x')], full_graph=True)

d = tempfile.mkdtemp(prefix='onnx_')
paddle.jit.save(sm, os.path.join(d, 'm'))
r = sp.run(['paddle2onnx', '--model_dir', d,
    '--model_filename', 'm.json', '--params_filename', 'm.pdiparams',
    '--save_file', os.path.join(d, 'out.onnx'),
    '--opset_version', '15', '--enable_onnx_checker', 'True'],
    capture_output=True, text=True, timeout=120)
if r.returncode != 0:
    print(f'  ONNX export FAILED: {r.stderr[-300:]}')
else:
    onnx_model = onnx.load(os.path.join(d, 'out.onnx'))
    op_types = {}
    for node in onnx_model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    print(f'  ONNX graph ops: {op_types}')

    sess = ort.InferenceSession(os.path.join(d, 'out.onnx'),
        providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None,
        {sess.get_inputs()[0].name: x.numpy()})[0]
    
    if np.any(np.isnan(onnx_out)):
        print(f'  ❌ ONNX output contains NaN!')
    else:
        diff = np.max(np.abs(eager - onnx_out))
        print(f'  ✅ Eager vs ONNX: max_diff={diff:.3e}')


# ── Test 2: Only LayerNorm (no MatMulAdd) ──
print("\nTest 2: LayerNorm(128) only (no Linear)")
class LNOnly(nn.Layer):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(128)
    def forward(self, x):
        return self.ln(x)

model2 = LNOnly()
model2.eval()
x2 = paddle.randn([4, 128])
eager2 = model2(x2).numpy()

sm2 = paddle.jit.to_static(model2,
    input_spec=[InputSpec([4, 128], 'float32', 'x')], full_graph=True)

d2 = tempfile.mkdtemp(prefix='onnx_')
paddle.jit.save(sm2, os.path.join(d2, 'm'))
r2 = sp.run(['paddle2onnx', '--model_dir', d2,
    '--model_filename', 'm.json', '--params_filename', 'm.pdiparams',
    '--save_file', os.path.join(d2, 'out.onnx'),
    '--opset_version', '15', '--enable_onnx_checker', 'True'],
    capture_output=True, text=True, timeout=120)
if r2.returncode != 0:
    print(f'  ONNX export FAILED: {r2.stderr[-300:]}')
else:
    sess2 = ort.InferenceSession(os.path.join(d2, 'out.onnx'),
        providers=['CPUExecutionProvider'])
    onnx_out2 = sess2.run(None,
        {sess2.get_inputs()[0].name: x2.numpy()})[0]
    if np.any(np.isnan(onnx_out2)):
        print(f'  ❌ ONNX output contains NaN!')
    else:
        diff2 = np.max(np.abs(eager2 - onnx_out2))
        print(f'  ✅ Eager vs ONNX: max_diff={diff2:.3e}')


# ── Test 3: Only MatMulAdd (no LayerNorm) ──
print("\nTest 3: MatMulAdd only (no LayerNorm)")
class MMAOnly(nn.Layer):
    def __init__(self):
        super().__init__()
        self.l1 = LinearViaMatMul(256, 128)
    def forward(self, x):
        return self.l1(x)

model3 = MMAOnly()
model3.eval()
x3 = paddle.randn([4, 256])
eager3 = model3(x3).numpy()

sm3 = paddle.jit.to_static(model3,
    input_spec=[InputSpec([4, 256], 'float32', 'x')], full_graph=True)

d3 = tempfile.mkdtemp(prefix='onnx_')
paddle.jit.save(sm3, os.path.join(d3, 'm'))
r3 = sp.run(['paddle2onnx', '--model_dir', d3,
    '--model_filename', 'm.json', '--params_filename', 'm.pdiparams',
    '--save_file', os.path.join(d3, 'out.onnx'),
    '--opset_version', '15', '--enable_onnx_checker', 'True'],
    capture_output=True, text=True, timeout=120)
if r3.returncode != 0:
    print(f'  ONNX export FAILED: {r3.stderr[-300:]}')
else:
    sess3 = ort.InferenceSession(os.path.join(d3, 'out.onnx'),
        providers=['CPUExecutionProvider'])
    onnx_out3 = sess3.run(None,
        {sess3.get_inputs()[0].name: x3.numpy()})[0]
    if np.any(np.isnan(onnx_out3)):
        print(f'  ❌ ONNX output contains NaN!')
    else:
        diff3 = np.max(np.abs(eager3 - onnx_out3))
        print(f'  ✅ Eager vs ONNX: max_diff={diff3:.3e}')
