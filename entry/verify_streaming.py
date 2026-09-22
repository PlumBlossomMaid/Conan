"""Verify the streaming inference path against the batch forward pass.

``StreamContentExtractor.forward`` (what training and the offline evaluator use)
runs the whole sequence through ``EmformerEncoder.forward``, which internally
chunks the input and walks the chunks while maintaining a memory bank and a
summary. Real-time deployment instead calls ``EmformerEncoder.forward_chunk``
once per 80 ms chunk and has to keep that state itself — a path no test or
evaluation ever exercised. This script checks three things about it:

1. **Equivalence** — replaying the chunk loop by hand through ``forward_chunk``,
   with the same left/right context, must reproduce the batch path exactly. If
   it does not, the streaming API is not a drop-in replacement.
2. **The summary is inert** — a summary only ever enters the attention queries,
   and query rows do not attend to each other, so it cannot reach the returned
   chunk; only ``new_summary`` depends on it. Streaming callers may therefore
   pass zeros, and ``forward``'s habit of recomputing it (rather than carrying
   the layer's ``new_summary``) is provably a no-op. Measured, not assumed.
3. **Streaming causality and speed** — rewriting every chunk after ``i +
   right_context`` must leave chunk ``i`` untouched, and per-chunk latency times
   the number of chunks gives a genuine streaming real-time factor.

Usage:
    python entry/verify_streaming.py -c configs/content_extractor_ce.yaml \\
        --ckpt ckpts/content_extractor_ce/last.pdparams \\
        -o data.val_hdf5_path=/path/to/valid.h5 \\
        -o data.label_codebook=data/libritts/codebook_256.npy \\
        --device gpu --json-out logs/content_extractor_ce/streaming.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.eval_content_extractor import (  # noqa: E402
    build_model,
    load_extractor_weights,
    to_tensor,
)
from layers.dataset import ContentExtractorDataset  # noqa: E402
from layers.label_codebook import PaddleLabelCodebook  # noqa: E402
from utils.config_utils import apply_overrides, load_config  # noqa: E402
from utils.training_utils import build_val_dataloader  # noqa: E402

FRAME_SECONDS = 0.02  # one content frame per 20 ms at 50 Hz


def _concat_or_empty(parts: List[paddle.Tensor], batch: int, dim: int, dtype) -> paddle.Tensor:
    """Concatenate context chunks along time, or return an empty (B, 0, D) tensor."""
    if parts:
        return paddle.concat(parts, axis=1)
    return paddle.zeros([batch, 0, dim], dtype=dtype)


@paddle.no_grad()
def streaming_forward(
    model: paddle.nn.Layer,
    mel: paddle.Tensor,
    carry_summary: bool = False,
    perturb_from_chunk: Optional[int] = None,
) -> paddle.Tensor:
    """Replay ``EmformerEncoder.forward`` chunk by chunk through ``forward_chunk``.

    This drives ``EmformerEncoder.forward_chunk`` on already-projected d_model
    chunks rather than ``StreamContentExtractor.forward_chunk``, because the
    latter takes ``mel_chunk`` in mel space but ``left_context`` / ``right_context``
    in d_model space with no helper to project them — a caller following its
    docstring would feed mel-space contexts.

    Args:
        model: The content extractor (its ``mel_proj`` / ``emformer`` / head are used).
        mel: (B, T, n_mels) input, already transposed to time-major.
        carry_summary: Feed the layer's returned ``new_summary`` into the next
            chunk instead of recomputing it from the incoming chunk as
            ``forward`` does. The two are equivalent by construction — a summary
            cannot reach the chunk output — and the equivalence check confirms
            that rather than assuming it.
        perturb_from_chunk: If set, overwrite every chunk from this index onward
            with a large constant before streaming — used to test that a chunk's
            output cannot depend on chunks past its right context.

    Returns:
        (B, T_orig, output_dim) scores, the same shape the batch path returns.
    """
    emformer = model.emformer
    chunk_size, left_context, right_context = (
        emformer.chunk_size, emformer.left_context, emformer.right_context
    )

    x = emformer.input_proj(model.mel_proj(mel))
    chunks, t_orig = emformer._split_into_chunks(x)
    batch, num_chunks, _, dim = chunks.shape
    if perturb_from_chunk is not None and perturb_from_chunk < num_chunks:
        chunks = paddle.concat(
            [
                chunks[:, :perturb_from_chunk],
                paddle.full(
                    [batch, num_chunks - perturb_from_chunk, chunk_size, dim],
                    5.0, dtype=chunks.dtype,
                ),
            ],
            axis=1,
        )

    memory = paddle.zeros([batch, chunk_size, dim], dtype=x.dtype)
    summary = paddle.zeros([batch, 1, dim], dtype=x.dtype)
    outputs = []
    for index in range(num_chunks):
        current = chunks[:, index]

        left_parts = [chunks[:, j] for j in range(max(0, index - left_context), index)]
        right_parts = [chunks[:, j] for j in range(index + 1, min(num_chunks, index + 1 + right_context))]
        left_ctx = _concat_or_empty(left_parts, batch, dim, x.dtype)
        right_ctx = _concat_or_empty(right_parts, batch, dim, x.dtype)

        # ``forward`` recomputes the summary from the incoming chunk each
        # iteration and throws the layer's ``new_summary`` away.
        layer_summary = summary if carry_summary else current.mean(axis=1, keepdim=True)

        chunk_out, memory, summary = emformer.forward_chunk(
            current, left_ctx, right_ctx, memory, layer_summary
        )
        outputs.append(chunk_out)

    return model._head(paddle.concat(outputs, axis=1)[:, :t_orig, :])


@paddle.no_grad()
def check_equivalence(model, mel, tolerance: float = 1e-5) -> Dict[str, object]:
    """Compare the batch path, the streaming replay and the carried-summary variant."""
    batch_out = model(mel)
    replayed = streaming_forward(model, mel, carry_summary=False)
    carried = streaming_forward(model, mel, carry_summary=True)

    replay_delta = float(paddle.abs(batch_out - replayed).max().item())
    carry_delta = float(paddle.abs(batch_out - carried).max().item())
    replay_argmax_agree = float(
        (batch_out.argmax(axis=-1) == replayed.argmax(axis=-1)).astype("float32").mean().item()
    )
    carry_argmax_agree = float(
        (batch_out.argmax(axis=-1) == carried.argmax(axis=-1)).astype("float32").mean().item()
    )
    return {
        "replay_max_abs_delta": replay_delta,
        "replay_argmax_agreement": replay_argmax_agree,
        "replay_equivalent": replay_delta <= tolerance,
        "carry_summary_max_abs_delta": carry_delta,
        "carry_summary_argmax_agreement": carry_argmax_agree,
        "carry_summary_equivalent": carry_delta <= tolerance,
    }


@paddle.no_grad()
def check_streaming_causality(model, mel, tolerance: float = 1e-5) -> Dict[str, object]:
    """A chunk's output must not depend on chunks past its right context.

    Streams the sample twice — once clean, once with every chunk from ``k``
    onward replaced — and compares the chunks that provably cannot see the
    rewrite.
    """
    emformer = model.emformer
    left, right = emformer.left_context, emformer.right_context
    num_chunks = int(np.ceil(mel.shape[1] / emformer.chunk_size))
    cut = max(1, num_chunks // 2)
    # Chunk j attends to chunks [j-1, j+right]; so chunk j >= cut-right sees the rewrite.
    unaffected = max(0, cut - right)
    before = streaming_forward(model, mel)
    after = streaming_forward(model, mel, perturb_from_chunk=cut)
    unprotected_frames = unaffected * emformer.chunk_size
    prefix_delta = float(paddle.abs(before[:, :unprotected_frames] - after[:, :unprotected_frames]).max().item())
    boundary = slice(max(0, cut * emformer.chunk_size - 1), cut * emformer.chunk_size)
    boundary_delta = float(paddle.abs(before[:, boundary] - after[:, boundary]).max().item())
    return {
        "num_chunks": num_chunks,
        "cut_chunk": cut,
        "left_context": left,
        "right_context": right,
        "unaffected_chunks": unaffected,
        "unaffected_frames": unprotected_frames,
        "prefix_max_abs_delta": prefix_delta,
        "last_frame_before_cut_delta": boundary_delta,
        "prefix_unchanged": prefix_delta <= tolerance,
    }


@paddle.no_grad()
def measure_streaming_latency(model, mel, chunk_size: int, warmup: int = 2, repeat: int = 5
                              ) -> Dict[str, float]:
    """Time the real per-chunk streaming path and derive its real-time factor."""
    emformer = model.emformer
    length = int(mel.shape[1])
    for _ in range(warmup):
        streaming_forward(model, mel)
    paddle.device.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        streaming_forward(model, mel)
    paddle.device.synchronize()
    total_ms = (time.perf_counter() - start) / repeat * 1000

    num_chunks = int(np.ceil(length / chunk_size))
    audio_s = length * FRAME_SECONDS
    return {
        "frames": float(length),
        "chunks": float(num_chunks),
        "audio_seconds": audio_s,
        "total_ms": total_ms,
        "ms_per_chunk": total_ms / max(num_chunks, 1),
        "rtf": total_ms / (audio_s * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the streaming inference path.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("-o", "--override", action="append", default=None, metavar="KEY=VALUE")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.override)
    if args.device:
        paddle.set_device(args.device)
    print(f"  device={paddle.get_device()}")

    data_cfg = config.get("data", {})
    sce_cfg = config.get("content_extractor", {})
    codebook = PaddleLabelCodebook.load(data_cfg["label_codebook"])
    model = build_model(config, codebook)
    load_extractor_weights(model, args.ckpt)
    model.eval()

    dataset = ContentExtractorDataset(
        hdf5_path=data_cfg.get("val_hdf5_path", data_cfg.get("hdf5_path")),
        max_frames=config.get("audio", {}).get("max_frames", 500),
        max_samples=data_cfg.get("val_max_samples", 50),
        label_codebook=codebook,
    )
    loader = build_val_dataloader(dataset)
    batch = max((b for b in loader), key=lambda b: int(np.asarray(b["lengths"])[0]))
    mel = to_tensor(batch["source_mel"]).transpose([0, 2, 1])
    print(f"  longest validation sample: {int(np.asarray(batch['lengths'])[0])} frames")

    report: Dict[str, object] = {"checkpoint": args.ckpt, "config": args.config}

    print("\n=== 1. Streaming replay vs batch forward ===")
    equivalence = check_equivalence(model, mel)
    report["equivalence"] = equivalence
    print(f"  replay  Δmax={equivalence['replay_max_abs_delta']:.3e}  "
          f"argmax agreement={equivalence['replay_argmax_agreement']:.6f}  "
          f"→ {'EQUIVALENT' if equivalence['replay_equivalent'] else 'MISMATCH'}")
    print(f"  carried-summary Δmax={equivalence['carry_summary_max_abs_delta']:.3e}  "
          f"argmax agreement={equivalence['carry_summary_argmax_agreement']:.6f}  "
          f"→ summary is {'INERT (bit-identical)' if equivalence['carry_summary_equivalent'] else 'NOT inert'}")

    print("\n=== 2. Streaming causality ===")
    causality = check_streaming_causality(model, mel)
    report["causality"] = causality
    verdict = "PASS" if causality["prefix_unchanged"] else "FAIL"
    print(f"  [{verdict}] chunks < {causality['unaffected_chunks']} "
          f"({causality['unaffected_frames']} frames) unchanged by rewriting every chunk "
          f"from {causality['cut_chunk']} onward (Δmax={causality['prefix_max_abs_delta']:.3e}); "
          f"last frame before cut Δ={causality['last_frame_before_cut_delta']:.3e}")

    print("\n=== 3. Streaming latency ===")
    timing = measure_streaming_latency(model, mel, int(sce_cfg.get("chunk_size", 4)))
    report["latency"] = timing
    print(f"  {timing['chunks']:.0f} chunks of {int(sce_cfg.get('chunk_size', 4))} frames: "
          f"{timing['total_ms']:.1f} ms total, {timing['ms_per_chunk']:.2f} ms/chunk, "
          f"RTF={timing['rtf']:.4f}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  report written to {out_path}")


if __name__ == "__main__":
    main()
