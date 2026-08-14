"""ONNX 逐层导出测试 — 夹逼法定位不支持算子。

策略：
1. 从最简单组件开始逐一测试 ONNX 导出（TimbreEncoder → PitchPredictor → Vocoder → MelDecoder）
2. 测试复杂组件（SCE/Emformer, ASE/CVQ, FusedConverter）
3. 当某组件失败时，对其子组件进行二分法测试，精确定位到具体算子

测试流程：
  model.eval() → paddle.jit.to_static() → paddle.jit.save() → paddle2onnx → onnx.checker

用法：
    python entry/test_onnx_export.py [--quick] [--sub]
    --quick: 只测组件级，不测子组件
    --sub:   只测子组件（用于二分法深入）
"""

import sys, os, tempfile, shutil, traceback
from pathlib import Path

import onnx
import numpy as np
import paddle
from paddle.static import InputSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["GLOG_minloglevel"] = "3"  # suppress PIR logging

# ── 层导入 ──
from layers.timbre_encoder import TimbreEncoder
from layers.causal_pitch_predictor import CausalPitchPredictor
from layers.causal_shuffle_vocoder import CausalShuffleVocoder
from layers.causal_mel_decoder import CausalMelDecoder
from layers.stream_content_extractor import StreamContentExtractor
from layers.emformer import EmformerEncoder, EmformerBlock
from layers.adaptive_style_encoder import AdaptiveStyleEncoder, PositionalEncoding
from layers.cvq import ClusteringVQ
from layers.causal_conv import CausalConv1D, CausalConvBlock
from layers.causal_shuffle_vocoder import CausalMRFResBlock


# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════
N_MELS = 80
CD = 512  # content_dim (SCE d_model, 注意这里是旧版 512，FusedConverter 也用这个)
TD = 256  # timbre_dim
SD = 64   # style_dim
T = 50    # 测试用帧数 (1s @ 50Hz)
CHUNK = 4


# ══════════════════════════════════════════════════════════════
# 测试函数
# ══════════════════════════════════════════════════════════════

def _to_static(model, input_specs, name="model"):
    """paddle.jit.to_static 编译。"""
    model.eval()
    sm = paddle.jit.to_static(model, input_spec=input_specs, full_graph=True)
    sm.eval()
    return sm


def _export_pir(static_model, export_dir, name):
    """paddle.jit.save → PIR 格式。"""
    model_dir = os.path.join(export_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    paddle.jit.save(static_model, os.path.join(model_dir, name))
    return model_dir


def _export_onnx(model_dir, name, save_path, verbose=False):
    """paddle2onnx 转换 + onnx 校验。"""
    import subprocess as sp

    cmd = [
        "paddle2onnx",
        "--model_dir", model_dir,
        "--model_filename", f"{name}.pdmodel",
        "--params_filename", f"{name}.pdiparams",
        "--save_file", save_path,
        "--opset_version", 15,
        "--enable_onnx_checker", "True",
    ]
    if verbose:
        cmd.append("--enable_verbose=True")

    r = sp.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        # 提取关键错误信息
        err = r.stderr[-2000:] if r.stderr else r.stdout[-2000:]
        return False, err
    return True, ""


def _verify_onnx(onnx_path):
    """onnx.checker 校验。"""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return True


# ══════════════════════════════════════════════════════════════
# 测试用例注册
# ══════════════════════════════════════════════════════════════

test_results = {}  # name -> True/False/error_msg


def test_layer(name, model_fn, input_specs, is_sub=False):
    """测试单个组件的 ONNX 导出流程。"""
    indent = "  " if is_sub else ""
    print(f"\n{indent}── {name} ──")

    try:
        model = model_fn()
        model.eval()
        sm = _to_static(model, input_specs, name)

        # 验证静态图能跑
        example = [paddle.randn(spec.shape) for spec in input_specs]
        out = sm(*example)
        out_np = out.numpy() if isinstance(out, paddle.Tensor) else out[0].numpy()
        print(f"{indent}  to_static ✅  out={out_np.shape}")

    except Exception as e:
        msg = f"to_static 失败: {type(e).__name__}: {e}"
        print(f"{indent}  to_static ❌  {msg}")
        test_results[name] = msg
        return

    # PIR export
    export_dir = tempfile.mkdtemp(prefix="onnx_test_")
    try:
        model_dir = _export_pir(sm, export_dir, name)
        pir_files = os.listdir(model_dir)
        print(f"{indent}  jit.save ✅  files={pir_files}")
    except Exception as e:
        msg = f"jit.save 失败: {type(e).__name__}: {e}"
        print(f"{indent}  jit.save ❌  {msg}")
        test_results[name] = msg
        shutil.rmtree(export_dir, ignore_errors=True)
        return

    # ONNX export
    onnx_path = os.path.join(export_dir, f"{name}.onnx")
    try:
        ok, err = _export_onnx(model_dir, name, onnx_path)
        if not ok:
            # 尝试提取算子信息
            print(f"{indent}  ONNX ❌  paddle2onnx failed")
            print(f"{indent}  Error excerpt:")
            for line in err.split("\n")[-15:]:
                print(f"{indent}    {line.strip()}")
            test_results[name] = f"paddle2onnx: {err.strip()[:500]}"
            shutil.rmtree(export_dir, ignore_errors=True)
            return

        # Verify ONNX
        _verify_onnx(onnx_path)
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"{indent}  ONNX ✅  {onnx_path}  ({size_mb:.1f} MB)")
        test_results[name] = True

    except Exception as e:
        msg = f"ONNX 验证失败: {type(e).__name__}: {e}"
        print(f"{indent}  ONNX ❌  {msg}")
        test_results[name] = msg

    shutil.rmtree(export_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# 测试用例
# ══════════════════════════════════════════════════════════════

# ── Level 1: 最简单组件 ──

def register_level1():
    """TimbreEncoder, CausalPitchPredictor, CausalShuffleVocoder"""

    test_layer(
        "TimbreEncoder",
        lambda: TimbreEncoder(n_mels=N_MELS, embed_dim=TD),
        [InputSpec([1, N_MELS, T * 2], "float32", "mel")],
    )

    test_layer(
        "CausalPitchPredictor",
        lambda: CausalPitchPredictor(content_dim=CD),
        [InputSpec([1, T, CD], "float32", "z_c")],
    )

    test_layer(
        "CausalShuffleVocoder",
        lambda: CausalShuffleVocoder(
            n_mels=N_MELS, upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
        ),
        [InputSpec([1, N_MELS, T], "float32", "mel")],
    )


# ── Level 2: 中等组件 ──

def register_level2():
    """CausalMelDecoder"""

    test_layer(
        "CausalMelDecoder",
        lambda: CausalMelDecoder(
            content_dim=CD, timbre_dim=TD, style_dim=SD, n_mels=N_MELS,
        ),
        [
            InputSpec([1, T, CD], "float32", "z_c"),
            InputSpec([1, TD], "float32", "z_t"),
            InputSpec([1, T, SD], "float32", "z_s"),
            InputSpec([1, T, 1], "float32", "f0"),
        ],
    )


# ── Level 3: 复杂组件（Transformer）──

def register_level3():
    """StreamContentExtractor (含 Emformer)"""

    test_layer(
        "StreamContentExtractor",
        lambda: StreamContentExtractor(
            input_dim=N_MELS, d_model=CD, nhead=8,
            num_layers=3, output_dim=256,  # 用 3 层快速测试
            chunk_size=CHUNK, left_context=1, right_context=2,
        ),
        [InputSpec([1, T, N_MELS], "float32", "mel")],
    )


# ── Level 4: ASE/CVQ ──

def register_level4():
    """AdaptiveStyleEncoder (含 CVQ + Align Attention)"""

    test_layer(
        "AdaptiveStyleEncoder",
        lambda: AdaptiveStyleEncoder(
            n_mels=N_MELS, style_dim=SD, code_dim=64, num_codes=128,
            timbre_dim=TD, content_dim=CD,
        ),
        [
            InputSpec([1, N_MELS, T * 2], "float32", "ref_mel"),
            InputSpec([1, T, CD], "float32", "z_c"),  # content
            InputSpec([1, TD], "float32", "z_t"),      # timbre
        ],
    )


# ── Level 5: Fused Converter ──

def register_level5():
    """FusedStreamingConverter — 全流水线合成"""

    test_layer(
        "FusedStreamingConverter",
        lambda: FusedConverterWrapper(),
        [
            InputSpec([1, T, N_MELS], "float32", "source_mel"),
            InputSpec([1, TD], "float32", "z_t"),
            InputSpec([1, T // 4 + 1, SD], "float32", "z_s"),
        ],
    )


# ══════════════════════════════════════════════════════════════
# 子组件测试（夹逼法用）
# ══════════════════════════════════════════════════════════════

def register_sub_components():
    """对复杂组件的子结构进行单独测试。"""

    # PositionalEncoding (ASE 内部)
    test_layer(
        "[sub] PositionalEncoding",
        lambda: PositionalEncoding(d_model=SD, max_len=512),
        [InputSpec([1, T, SD], "float32", "x")],
        is_sub=True,
    )

    # ClusteringVQ 纯前向（eval mode，无 loss）
    test_layer(
        "[sub] ClusteringVQ",
        lambda: ClusteringVQ(code_dim=64, num_codes=128),
        [InputSpec([1, 64, T], "float32", "z")],
        is_sub=True,
    )

    # EmformerBlock
    test_layer(
        "[sub] EmformerBlock",
        lambda: EmformerBlock(
            d_model=CD, nhead=8, dim_feedforward=2048,
            left_context=1, right_context=2,
        ),
        [
            InputSpec([1, CHUNK, CD], "float32", "chunk"),
            InputSpec([1, CHUNK, CD], "float32", "left_context"),
            InputSpec([1, 2 * CHUNK, CD], "float32", "right_context"),
            InputSpec([1, CHUNK, CD], "float32", "memory"),
            InputSpec([1, 1, CD], "float32", "summary"),
        ],
        is_sub=True,
    )

    # EmformerEncoder（完整 forward）
    test_layer(
        "[sub] EmformerEncoder",
        lambda: EmformerEncoder(
            d_model=CD, nhead=8, num_layers=3,
            left_context=1, right_context=2, chunk_size=CHUNK,
        ),
        [InputSpec([1, T, CD], "float32", "x")],
        is_sub=True,
    )

    # CausalConv1D alone
    test_layer(
        "[sub] CausalConv1D",
        lambda: CausalConv1D(CD, CD, 5, dilation=2, weight_norm=True),
        [InputSpec([1, CD, T], "float32", "x")],
        is_sub=True,
    )

    # CausalConvBlock
    test_layer(
        "[sub] CausalConvBlock",
        lambda: CausalConvBlock(CD, CD // 2, 3, stride=2, dilation=1, use_act=True),
        [InputSpec([1, CD, T], "float32", "x")],
        is_sub=True,
    )

    # CausalMRFResBlock
    test_layer(
        "[sub] CausalMRFResBlock",
        lambda: CausalMRFResBlock(512, 3, [1, 3, 5]),
        [InputSpec([1, 512, T], "float32", "x")],
        is_sub=True,
    )


# ══════════════════════════════════════════════════════════════
# FusedConverter 包装（修复 num_classes 参数问题）
# ══════════════════════════════════════════════════════════════

class FusedConverterWrapper(paddle.nn.Layer):
    """修复 FusedStreamingConverter num_classes 参数不匹配的问题。"""

    def __init__(self):
        super().__init__()
        # 改用 output_dim=256 (SCE 当前版本)
        self.content_dim = CD

        self.content_extractor = StreamContentExtractor(
            input_dim=N_MELS, d_model=CD, nhead=8,
            num_layers=3, output_dim=CD,  # output_dim = content_dim
            chunk_size=CHUNK, left_context=1, right_context=2,
        )

        align_dim = CD + TD
        self.align_q_proj = paddle.nn.Linear(align_dim, SD)
        self.align_k_proj = paddle.nn.Linear(SD, SD)
        self.align_v_proj = paddle.nn.Linear(SD, SD)
        self.align_out = paddle.nn.Linear(SD, SD)

        self.pitch_predictor = CausalPitchPredictor(content_dim=CD)
        self.mel_decoder = CausalMelDecoder(
            content_dim=CD, timbre_dim=TD, style_dim=SD, n_mels=N_MELS,
        )
        self.vocoder = CausalShuffleVocoder(
            n_mels=N_MELS, upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
        )

    def forward(self, source_mel, z_t, z_s):
        B, T, _ = source_mel.shape
        z_c = self.content_extractor(source_mel)  # (B, T, CD)

        z_t_b = z_t.unsqueeze(1).expand([-1, T, -1])
        z_ct = paddle.concat([z_c, z_t_b], axis=-1)

        Q = self.align_q_proj(z_ct)
        K = self.align_k_proj(z_s)
        V = self.align_v_proj(z_s)
        attn = paddle.matmul(Q, K, transpose_y=True) / (SD ** 0.5)
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        z_s_aligned = paddle.matmul(attn, V)
        z_s_aligned = self.align_out(z_s_aligned)

        f0 = self.pitch_predictor(z_c)
        mel = self.mel_decoder(z_c, z_t, z_s_aligned, f0)
        audio = self.vocoder(mel)
        return audio


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ONNX 逐层导出测试")
    parser.add_argument("--quick", action="store_true", help="只测组件级")
    parser.add_argument("--sub", action="store_true", help="只测子组件")
    parser.add_argument("--specific", type=str, default=None,
                        help="只测特定组件（逗号分隔，如 TimbreEncoder,CausalPitchPredictor）")
    args = parser.parse_args()

    print(f"Paddle {paddle.__version__} | CUDA: {paddle.is_compiled_with_cuda()}")
    print(f"T={T} frames (≈{T*320/16000:.1f}s @16kHz)")
    print("=" * 70)
    print("  ONNX 逐层导出测试 — 夹逼法定位不支持算子")
    print("=" * 70)

    # 按复杂度从低到高注册
    if args.sub:
        register_sub_components()
    elif args.quick:
        register_level1()
        register_level2()
    else:
        register_level1()  # 简单
        register_level2()  # 中等
        register_level3()  # 复杂
        register_level4()  # 复杂
        register_level5()  # 全流水线
        register_sub_components()  # 子组件

    # 如果有 --specific，只运行匹配的
    if args.specific:
        targets = set(args.specific.split(","))
        for k in list(test_results.keys()):
            if k not in targets:
                del test_results[k]

    # ═══ 汇总 ═══
    print("\n" + "=" * 70)
    print("  汇总报告")
    print("=" * 70)

    passed = []
    failed = []
    for name, result in test_results.items():
        if result is True:
            passed.append(name)
        else:
            failed.append((name, str(result)[:200]))

    if passed:
        print(f"\n✅ 成功 ({len(passed)}):")
        for n in passed:
            print(f"   ✅ {n}")

    if failed:
        print(f"\n❌ 失败 ({len(failed)}):")
        for n, err in failed:
            print(f"   ❌ {n}")
            # 提取算子名
            op_hints = []
            for line in err.split("\\n"):
                if "not support" in line.lower() or "unsupported" in line.lower() or "not implemented" in line.lower():
                    op_hints.append(line.strip())
                if "op=" in line.lower() or "operator" in line.lower():
                    op_hints.append(line.strip())
            if op_hints:
                print(f"      算子线索:")
                for h in op_hints[:5]:
                    print(f"        {h}")
            print(f"      错误: {err[:200]}")
    else:
        print("\n🎉 所有组件 ONNX 导出成功！")


if __name__ == "__main__":
    main()
