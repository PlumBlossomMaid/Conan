"""Test ONNX export for all Conan components using PIR format (full_graph=True).

The process:
1. paddle.jit.to_static(full_graph=True) → PIR static model
2. paddle.jit.save() → produces .json + .pdiparams
3. paddle2onnx --model_filename xxx.json → ONNX

This works for most ops (conv2d, leaky_relu, layer_norm, etc.) but
fails for some (linear_v2, group_norm).
"""
import sys, os, tempfile, subprocess as sp, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import paddle
from paddle.static import InputSpec

from layers.timbre_encoder import TimbreEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_shuffle_vocoder import CausalShuffleVocoder
from layers.causal_mel_decoder import CausalMelDecoder
from layers.stream_content_extractor import StreamContentExtractor
from layers.adaptive_style_encoder import AdaptiveStyleEncoder

N_MELS, CD, TD, SD, T, CHUNK = 80, 512, 256, 64, 50, 4


def test_onnx(name, model_fn, specs):
    """Test PIR→ONNX conversion for one component."""
    model = model_fn()
    model.eval()
    
    # 1. to_static (PIR)
    try:
        sm = paddle.jit.to_static(model, input_spec=specs, full_graph=True)
    except Exception as e:
        return f"to_static FAILED: {e}", []
    
    # 2. jit.save (PIR format: .json + .pdiparams)
    d = tempfile.mkdtemp(prefix='onnx_')
    spath = os.path.join(d, name)
    try:
        paddle.jit.save(sm, spath)
    except Exception as e:
        return f"jit.save FAILED: {e}", []
    
    files = os.listdir(d)
    has_json = any(f.endswith('.json') for f in files)
    if not has_json:
        return "No .json file (not PIR format)", files
    
    # 3. Extract PIR ops for reporting
    json_path = os.path.join(d, f'{name}.json')
    with open(json_path) as f:
        data = json.load(f)
    ops_list = data.get('program',{}).get('regions',[{}])[0].get('blocks',[{}])[0].get('ops',[])
    pir_ops = sorted(set(
        tag.split('.', 1)[1] for op in ops_list for tag in [op.get('#','')] 
        if '.' in tag and tag.split('.',1)[1] not in 
        ('data','assign','full_int_array','full','fetch','set_parameter','builtin')
    ))
    
    # 4. paddle2onnx
    cmd = ['paddle2onnx', '--model_dir', d,
           '--model_filename', f'{name}.json',
           '--params_filename', f'{name}.pdiparams',
           '--save_file', os.path.join(d, f'{name}.onnx'),
           '--opset_version', '15', '--enable_onnx_checker', 'True']
    r = sp.run(cmd, capture_output=True, text=True, timeout=120)
    
    onnx_path = os.path.join(d, f'{name}.onnx')
    if r.returncode == 0 and os.path.exists(onnx_path):
        size = os.path.getsize(onnx_path) / 1024
        return True, pir_ops, size
    else:
        # Try to identify the failing operator
        err_msg = r.stderr[-500:] if r.stderr else r.stdout[-500:]
        hint_ops = []
        for op in pir_ops:
            if op.lower() in err_msg.lower():
                hint_ops.append(op)
        return f"FAILED: likely {hint_ops or 'unknown'}", pir_ops


# ── Define all test cases ──
tests = [
    ("TimbreEncoder",
     lambda: TimbreEncoder(n_mels=N_MELS, embed_dim=TD),
     [InputSpec([1, N_MELS, T*2], 'float32', 'mel')]),
    
    ("CausalPitchPredictor",
     lambda: CausalPitchPredictor(content_dim=CD),
     [InputSpec([1, T, CD], 'float32', 'z_c')]),
    
    ("CausalShuffleVocoder",
     lambda: CausalShuffleVocoder(n_mels=N_MELS, upsample_rates=[8,8,2,2], upsample_initial_channel=512),
     [InputSpec([1, N_MELS, T], 'float32', 'mel')]),
    
    ("CausalMelDecoder",
     lambda: CausalMelDecoder(content_dim=CD, timbre_dim=TD, style_dim=SD, n_mels=N_MELS),
     [InputSpec([1, T, CD], 'float32', 'z_c'),
      InputSpec([1, TD], 'float32', 'z_t'),
      InputSpec([1, T, SD], 'float32', 'z_s'),
      InputSpec([1, T, 1], 'float32', 'f0')]),
    
    ("StreamContentExtractor",
     lambda: StreamContentExtractor(input_dim=N_MELS, d_model=CD, nhead=8, num_layers=3, output_dim=256, chunk_size=CHUNK, left_context=1, right_context=2),
     [InputSpec([1, T, N_MELS], 'float32', 'mel')]),
    
    ("AdaptiveStyleEncoder",
     lambda: AdaptiveStyleEncoder(n_mels=N_MELS, style_dim=SD, code_dim=64, num_codes=128, timbre_dim=TD, content_dim=CD),
     [InputSpec([1, N_MELS, T*2], 'float32', 'ref_mel'),
      InputSpec([1, T, CD], 'float32', 'z_c'),
      InputSpec([1, TD], 'float32', 'z_t')]),
]

# ── Run ──
print(f"Paddle {paddle.__version__} | paddle2onnx 2.1.0")
print("=" * 70)
print(f"Testing {len(tests)} components with PIR format → ONNX")
print("=" * 70)

all_results = []
for name, fn, specs in tests:
    result = test_onnx(name, fn, specs)
    
    if result[0] is True:
        _, pir_ops, size_kb = result
        print(f"  ✅ {name:30s} ONNX {size_kb:6.1f}KB  ops={len(pir_ops)}")
        all_results.append((name, True, pir_ops, size_kb))
    else:
        msg, pir_ops = result
        print(f"  ❌ {name:30s} {msg}")
        all_results.append((name, False, pir_ops, msg))

# ── Summary ──
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_ops = set()
for name, ok, pir_ops, *_ in all_results:
    all_ops.update(pir_ops)

print(f"\nAll PIR ops across components ({len(all_ops)} total):")
for op in sorted(all_ops):
    comps = [n for n, ok, ops, *_ in all_results if op in ops]
    statuses = [('✅' if ok else '❌') for n, ok, ops, *_ in all_results if op in ops]
    # Only unique
    failed_comps = [n for n, ok, ops, *_ in all_results if op in ops and not ok]
    note = " ⚠️  IN FAILED COMPONENT" if failed_comps else ""
    print(f"  {op:25s}  →  {', '.join(f'{s} {n}' for s, n in zip(statuses, comps))}{note}")

# Identify which ops from successful exports are "safe"
safe_ops = set()
for name, ok, pir_ops, *_ in all_results:
    if ok:
        safe_ops.update(pir_ops)

failed_comps = [(n, msg) for n, ok, po, *m in all_results if not ok for msg in [m[0]]]
if failed_comps:
    print(f"\n❌ Failed components ({len(failed_comps)}):")
    for n, msg in failed_comps:
        print(f"  ❌ {n}")
        # Show which ops are unique to this component
        comp_ops = next(po for name2, ok, po, *_ in all_results if name2 == n)
        unsafe = [o for o in comp_ops if o not in safe_ops]
        if unsafe:
            print(f"     Suspicious ops: {unsafe}")
else:
    print(f"\n✅ All {len(all_results)} components exported successfully!")
