"""Debug numerical mismatch between PIR and ONNX Runtime.

TimbreEncoder_fixed exports to ONNX but outputs differ (max_diff=2.67).
This script identifies the root cause.
"""
import sys, os, tempfile, subprocess as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import numpy as np
import onnx
import onnxruntime as ort
import paddle
from paddle.static import InputSpec
import paddle.nn as nn

from layers.causal_conv import CausalConvBlock

N_MELS, TD = 80, 256


class LinearViaMatMul(nn.Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = self.create_parameter(
            [in_features, out_features],
            default_initializer=nn.initializer.XavierUniform())
        self.b = self.create_parameter(
            [out_features],
            default_initializer=nn.initializer.Constant(0.0))
    def forward(self, x):
        return paddle.matmul(x, self.w) + self.b


class TimbreEncoderFixed(nn.Layer):
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


def debug_outputs():
    """Trace outputs at each stage to find where PIR and ONNX diverge."""
    model = TimbreEncoderFixed()
    model.eval()
    
    # Use fixed seed for reproducibility
    paddle.seed(42)
    mel = paddle.randn([1, N_MELS, 100])
    mel_np = mel.numpy()
    
    # ── Eager ──
    eager_out = model(mel)
    
    # ── PIR ──
    sm = paddle.jit.to_static(model, 
        input_spec=[InputSpec([1, N_MELS, 100], 'float32', 'mel')],
        full_graph=True)
    pir_out = sm(paddle.to_tensor(mel_np))
    
    # ── ONNX export ──
    d = tempfile.mkdtemp(prefix='onnx_')
    paddle.jit.save(sm, os.path.join(d, 'm'))
    
    r = sp.run(['paddle2onnx', '--model_dir', d,
           '--model_filename', 'm.json', '--params_filename', 'm.pdiparams',
           '--save_file', os.path.join(d, 'out.onnx'),
           '--opset_version', '15', '--enable_onnx_checker', 'True'],
           capture_output=True, text=True, timeout=120)
    
    if r.returncode != 0:
        print(f"ONNX export FAILED: {r.stderr[-500:]}")
        return
    
    onnx_path = os.path.join(d, 'out.onnx')
    print(f"ONNX export OK: {os.path.getsize(onnx_path)/1024:.1f}KB")
    
    # ── ONNX Runtime ──
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {onnx_input_name: mel_np})[0]
    
    print(f"\n── Comparison ──")
    print(f"Eager:  min={eager_out.numpy().min():.4f}  max={eager_out.numpy().max():.4f}  mean={eager_out.numpy().mean():.4f}")
    print(f"PIR:    min={pir_out.numpy().min():.4f}  max={pir_out.numpy().max():.4f}  mean={pir_out.numpy().mean():.4f}")
    print(f"ONNX:   min={onnx_out.min():.4f}  max={onnx_out.max():.4f}  mean={onnx_out.mean():.4f}")
    
    diff_ep = np.max(np.abs(eager_out.numpy() - pir_out.numpy()))
    diff_po = np.max(np.abs(pir_out.numpy() - onnx_out))
    diff_eo = np.max(np.abs(eager_out.numpy() - onnx_out))
    print(f"\nEager vs PIR:  max_diff={diff_ep:.3e}")
    print(f"PIR vs ONNX:   max_diff={diff_po:.3e}")
    print(f"Eager vs ONNX: max_diff={diff_eo:.3e}")
    
    # ── Check if weights were loaded correctly ──
    # Compare the model's parameters with what ONNX has
    print(f"\n── Per-stage analysis ──")
    
    # Run intermediate outputs through both PIR and ONNX
    # This requires modifying the model, so let's just compare
    # the ONNX graph structure
    onnx_model = onnx.load(onnx_path)
    
    # Count ops
    op_counts = {}
    for node in onnx_model.graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    print(f"ONNX graph ops: {op_counts}")
    
    # Check inputs/outputs
    print(f"ONNX inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"ONNX outputs: {[o.name for o in onnx_model.graph.output]}")
    
    # Check if there's a constant with wrong dimensions
    for init in onnx_model.graph.initializer:
        if 'w' in init.name or 'W' in init.name or 'weight' in init.name.lower():
            dims = list(init.dims)
            if 'matmul' in init.name.lower() or (len(dims) == 2 and dims[0] > 10):
                print(f"  Weight '{init.name}': dims={dims}")
    
    print(f"\n── Test with nn.Linear (original version) for comparison ──")
    # The original TimbreEncoder with nn.Linear might also export if we remove 
    # other issues. Let's check what weights look like.
    from layers.timbre_encoder import TimbreEncoder as TimbreEncoderOrig
    model_orig = TimbreEncoderOrig()
    model_orig.eval()
    
    # Check the linear weight shapes
    for name, param in model_orig.named_parameters():
        if 'proj' in name:
            print(f"  Original: {name}: shape={param.shape}")
    for name, param in model.named_parameters():
        if 'proj' in name:
            print(f"  Fixed:    {name}: shape={param.shape}")


if __name__ == '__main__':
    debug_outputs()
