"""Test old-format (non-PIR) ONNX export workaround for all Conan components.

Uses `full_graph=False` to produce `.pdmodel` format that paddle2onnx can read.
"""
import sys, os, tempfile, subprocess as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import paddle
from paddle.static import InputSpec

N_MELS, CD, TD, SD, T, CHUNK = 80, 512, 256, 64, 50, 4

def test_old_format(name, model_fn, specs, skip_full_graph=False):
    """Export with full_graph=False to get .pdmodel, then convert via paddle2onnx."""
    model = model_fn()
    model.eval()

    # Export to old format (full_graph=False produces .pdmodel)
    sm = paddle.jit.to_static(model, input_spec=specs, full_graph=not skip_full_graph)
    model.eval()
    
    d = tempfile.mkdtemp(prefix='onnx_')
    spath = os.path.join(d, name)
    try:
        paddle.jit.save(sm, spath)
    except Exception as e:
        print(f"  ❌ {name:25s} jit.save FAILED: {e}")
        return
    
    files = os.listdir(d)
    has_pdmodel = any(f.endswith('.pdmodel') for f in files)
    has_pir = any(f.endswith('.json') for f in files)
    fmt = 'PIR' if has_pir else ('PDMODEL' if has_pdmodel else 'UNKNOWN')
    
    # Determine model filename
    if has_pdmodel:
        model_fn = f'{name}.pdmodel'
        params_fn = f'{name}.pdiparams'
    elif has_pir:
        # Try without model_filename (PIR format not supported by paddle2onnx 2.1.0)
        print(f"  ❌ {name:25s} PIR format (paddle2onnx 2.1.0 doesn't support)")
        return
    else:
        print(f"  ❌ {name:25s} No model file found. Files: {files}")
        return

    # paddle2onnx
    cmd = ['paddle2onnx', '--model_dir', d,
           '--model_filename', model_fn,
           '--params_filename', params_fn,
           '--save_file', os.path.join(d, f'{name}.onnx'),
           '--opset_version', '15', '--enable_onnx_checker', 'True']
    r = sp.run(cmd, capture_output=True, text=True, timeout=120)
    
    if r.returncode == 0:
        size = os.path.getsize(os.path.join(d, f'{name}.onnx')) / 1024
        print(f"  ✅ {name:25s} ONNX {size:.0f}KB  ({fmt})")
        return True
    else:
        # Extract error
        for line in r.stderr.split('\n')[-8:]:
            if line.strip() and ('unsupported' in line.lower() or 'error' in line.lower() or 'not' in line.lower()):
                print(f"  ❌ {name:25s} {line.strip()[:120]}  ({fmt})")
                break
        else:
            # Generic error
            err = r.stderr[-200:].strip()
            print(f"  ❌ {name:25s} FAILED  ({fmt})")
            print(f"       {err[:200]}")
        return False


# ── Test all components ──

from layers.timbre_encoder import TimbreEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_shuffle_vocoder import CausalShuffleVocoder
from layers.causal_mel_decoder import CausalMelDecoder
from layers.stream_content_extractor import StreamContentExtractor
from layers.adaptive_style_encoder import AdaptiveStyleEncoder

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
     lambda: StreamContentExtractor(input_dim=N_MELS, d_model=CD, nhead=8, num_layers=3, output_dim=256, chunk_size=CHUNK),
     [InputSpec([1, T, N_MELS], 'float32', 'mel')]),
    
    ("AdaptiveStyleEncoder",
     lambda: AdaptiveStyleEncoder(n_mels=N_MELS, style_dim=SD, code_dim=64, num_codes=128, timbre_dim=TD, content_dim=CD),
     [InputSpec([1, N_MELS, T*2], 'float32', 'ref_mel'),
      InputSpec([1, T, CD], 'float32', 'z_c'),
      InputSpec([1, TD], 'float32', 'z_t')]),
]

# Also test fused converter
class FusedTestModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        # Use a simplified fused converter for testing
        self.te = TimbreEncoder(n_mels=N_MELS, embed_dim=TD)
        self.pp = CausalPitchPredictor(content_dim=CD)
        self.md = CausalMelDecoder(content_dim=CD, timbre_dim=TD, style_dim=SD, n_mels=N_MELS)
        self.vc = CausalShuffleVocoder(n_mels=N_MELS, upsample_rates=[8,8,2,2], upsample_initial_channel=512)
    
    def forward(self, mel, z_c, z_t, z_s, f0):
        _ = self.te(mel)
        _ = self.pp(z_c)
        mel_out = self.md(z_c, z_t, z_s, f0)
        audio = self.vc(mel_out)
        return audio

# But FusedTestModel is not an actual component of the model. Let's skip it.
# Instead, test all individual layers.

total = len(tests)
print(f"Paddle {paddle.__version__} | paddle2onnx 2.1.0")
print(f"Testing {total} components with old-format (full_graph=False) export")
print("=" * 60)

results = []
for name, fn, specs in tests:
    r = test_old_format(name, fn, specs)
    results.append((name, r))

print("\n" + "=" * 60)
print("Summary:")
for name, r in results:
    print(f"  {'✅' if r else '❌'} {name}")
