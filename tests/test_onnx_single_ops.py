"""Test individual PIR operators for ONNX export compatibility."""
import sys, os, tempfile, subprocess as sp, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import paddle
from paddle.static import InputSpec


# Define all test model classes at module level so inspect.getsource works

class LinearModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(64, 128)
    def forward(self, x):
        return self.linear(x)

class Conv1DModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv1D(8, 16, 3)
    def forward(self, x):
        return self.conv(x)

class LeakyReLUModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.act = paddle.nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.act(x)

class LayerNormModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.norm = paddle.nn.LayerNorm(64)
    def forward(self, x):
        return self.norm(x)

class Pad1DModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.pad = paddle.nn.Pad1D([2, 0])
    def forward(self, x):
        return self.pad(x)

class AdaptiveAvgPoolModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.pool = paddle.nn.AdaptiveAvgPool1D(1)
    def forward(self, x):
        return self.pool(x)

class FlattenModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.flatten = paddle.nn.Flatten()
    def forward(self, x):
        return self.flatten(x)

class TanhModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.act = paddle.nn.Tanh()
    def forward(self, x):
        return self.act(x)

class GELUModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.act = paddle.nn.GELU()
    def forward(self, x):
        return self.act(x)

class SoftmaxModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.act = paddle.nn.Softmax()
    def forward(self, x):
        return self.act(x)

class MatMulModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([50, 32])
    def forward(self, x):
        return paddle.matmul(x, self.w)

class GroupNormModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.gn = paddle.nn.GroupNorm(1, 64)
    def forward(self, x):
        return self.gn(x)

class PixelShuffleModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.ps = paddle.nn.PixelShuffle(2)
    def forward(self, x):
        return self.ps(x)

class MeanModel(paddle.nn.Layer):
    def forward(self, x):
        return x.mean(axis=-1, keepdim=True)

class ExpandModel(paddle.nn.Layer):
    def forward(self, x):
        return x.unsqueeze(1).expand([-1, 50, -1])

class ConcatModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([50, 64])
    def forward(self, x):
        return paddle.concat([x, self.w.expand([1, 50, 64])], axis=-1)

class ArgmaxModel(paddle.nn.Layer):
    def forward(self, x):
        return paddle.argmax(x, axis=-1)

class SigmoidModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.act = paddle.nn.Sigmoid()
    def forward(self, x):
        return self.act(x)

class DropoutModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.d = paddle.nn.Dropout(0.1)
    def forward(self, x):
        return self.d(x)

class EmbeddingModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.emb = paddle.nn.Embedding(100, 64)
    def forward(self, x):
        return self.emb(x)


def test(name, model_fn, spec, *extra_specs):
    specs = [spec] + list(extra_specs)
    model = model_fn()
    model.eval()
    
    try:
        sm = paddle.jit.to_static(model, input_spec=specs, full_graph=True)
    except Exception as e:
        print(f"  ❌ {name:20s} to_static FAILED: {e}")
        return False

    d = tempfile.mkdtemp(prefix='onnx_')
    spath = os.path.join(d, 'm')
    try:
        paddle.jit.save(sm, spath)
    except Exception as e:
        print(f"  ❌ {name:20s} jit.save FAILED: {e}")
        return False

    # Read the PIR ops
    with open(spath + '.json') as f:
        data = json.load(f)
    ops_list = data.get('program',{}).get('regions',[{}])[0].get('blocks',[{}])[0].get('ops',[])
    pir_ops = set()
    for op in ops_list:
        tag = op.get('#', '')
        if '.' in tag:
            pir_ops.add(tag.split('.', 1)[1])

    # Try paddle2onnx
    cmd = ['paddle2onnx', '--model_dir', d,
           '--model_filename', 'm.json',
           '--params_filename', 'm.pdiparams',
           '--save_file', os.path.join(d, 'out.onnx'),
           '--opset_version', '15', '--enable_onnx_checker', 'True']
    r = sp.run(cmd, capture_output=True, text=True, timeout=120)
    
    if r.returncode == 0:
        size = os.path.getsize(os.path.join(d, 'out.onnx')) / 1024
        print(f"  ✅ {name:20s} ONNX {size:.0f}KB  pir_ops={sorted(pir_ops)}")
        return True
    else:
        err = r.stderr
        # Extract hint about which op
        hint = ''
        for pir_op in pir_ops:
            if pir_op in err.lower() or pir_op.replace('_v2','') in err.lower() or pir_op.replace('_v2','_v1') in err.lower():
                hint = f' (hint: {pir_op})'
                break
        print(f"  ❌ {name:20s} paddle2onnx FAILED{hint}")
        for line in err.split('\n')[-5:]:
            if line.strip():
                print(f"       {line.strip()[:120]}")
        return False


print(f"Paddle {paddle.__version__} | paddle2onnx 2.1.0")
print("=" * 60)
print("Single-operator ONNX test")
print("=" * 60)

# Core ops from TimbreEncoder
test("conv2d", Conv1DModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("leaky_relu", LeakyReLUModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("linear_v2", LinearModel, InputSpec([1, 50, 64], 'float32', 'x'))
test("layer_norm", LayerNormModel, InputSpec([1, 50, 64], 'float32', 'x'))
test("pad3d", Pad1DModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("pool2d", AdaptiveAvgPoolModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("flatten/reshape", FlattenModel, InputSpec([1, 8, 50], 'float32', 'x'))

# Additional ops needed for other components
test("tanh", TanhModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("gelu", GELUModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("softmax", SoftmaxModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("matmul_v2", MatMulModel, InputSpec([1, 50, 50], 'float32', 'x'))
test("group_norm", GroupNormModel, InputSpec([1, 64, 50], 'float32', 'x'))
test("pixel_shuffle", PixelShuffleModel, InputSpec([1, 8, 50, 50], 'float32', 'x'))
test("sigmoid", SigmoidModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("dropout", DropoutModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("embedding", EmbeddingModel, InputSpec([1, 50], 'int64', 'x'))
test("mean", MeanModel, InputSpec([1, 8, 50], 'float32', 'x'))
test("expand_as", ExpandModel, InputSpec([1, 8], 'float32', 'x'))
test("concat", ConcatModel, InputSpec([1, 50, 64], 'float32', 'x'))
test("argmax", ArgmaxModel, InputSpec([1, 50, 100], 'float32', 'x'))
