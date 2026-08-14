"""Dump PIR operator list for each Conan component."""
import sys, os, tempfile, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['GLOG_minloglevel'] = '3'

import paddle
from paddle.static import InputSpec

from layers.timbre_encoder import TimbreEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_shuffle_vocoder import CausalShuffleVocoder
from layers.causal_mel_decoder import CausalMelDecoder
from layers.stream_content_extractor import StreamContentExtractor
from layers.emformer import EmformerEncoder, EmformerBlock
from layers.adaptive_style_encoder import AdaptiveStyleEncoder
from layers.cvq import ClusteringVQ

N_MELS, CD, TD, SD, T, CHUNK = 80, 512, 256, 64, 50, 4


def dump_ops(name, model_fn, specs):
    model = model_fn()
    model.eval()
    sm = paddle.jit.to_static(model, input_spec=specs, full_graph=True)
    d = tempfile.mkdtemp(prefix='pir_')
    os.makedirs(os.path.join(d, name), exist_ok=True)
    paddle.jit.save(sm, os.path.join(d, name))
    with open(os.path.join(d, name, f'{name}.json')) as f:
        data = json.load(f)
    text = json.dumps(data)
    ops = sorted(set(re.findall(r'"#"\s*:\s*"(pd_op\.[^"]+)"', text)))
    # also find combined ops
    for m in re.finditer(r'"#"\s*:\s*"([^"]*pd_op\.combined[^"]*)"', text):
        ops.append(m.group(1))
    print(f'\n{"="*60}')
    ops = sorted(set(ops))
    print(f'{name} — {len(ops)} unique ops:')
    for op in ops:
        print(f'  {op}')


# Level 1: simple
dump_ops('TimbreEncoder',
    lambda: TimbreEncoder(n_mels=N_MELS, embed_dim=TD),
    [InputSpec([1, N_MELS, T * 2], 'float32', 'mel')])

dump_ops('CausalPitchPredictor',
    lambda: CausalPitchPredictor(content_dim=CD),
    [InputSpec([1, T, CD], 'float32', 'z_c')])

dump_ops('CausalShuffleVocoder',
    lambda: CausalShuffleVocoder(n_mels=N_MELS, upsample_rates=[8,8,2,2], upsample_initial_channel=512),
    [InputSpec([1, N_MELS, T], 'float32', 'mel')])

dump_ops('CausalMelDecoder',
    lambda: CausalMelDecoder(content_dim=CD, timbre_dim=TD, style_dim=SD, n_mels=N_MELS),
    [InputSpec([1, T, CD], 'float32', 'z_c'),
     InputSpec([1, TD], 'float32', 'z_t'),
     InputSpec([1, T, SD], 'float32', 'z_s'),
     InputSpec([1, T, 1], 'float32', 'f0')])

# Level 3: SCE
dump_ops('StreamContentExtractor',
    lambda: StreamContentExtractor(
        input_dim=N_MELS, d_model=CD, nhead=8, num_layers=3, output_dim=256,
        chunk_size=CHUNK, left_context=1, right_context=2),
    [InputSpec([1, T, N_MELS], 'float32', 'mel')])

# Level 4: ASE
dump_ops('AdaptiveStyleEncoder',
    lambda: AdaptiveStyleEncoder(
        n_mels=N_MELS, style_dim=SD, code_dim=64, num_codes=128,
        timbre_dim=TD, content_dim=CD),
    [InputSpec([1, N_MELS, T * 2], 'float32', 'ref_mel'),
     InputSpec([1, T, CD], 'float32', 'z_c'),
     InputSpec([1, TD], 'float32', 'z_t')])
