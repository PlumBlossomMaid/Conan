import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import paddle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.hubert import HubertTeacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("pdparams_path", type=Path)
    parser.add_argument("--samples", type=int, default=16000)
    args = parser.parse_args()

    paddle.set_device("cpu")
    rng = np.random.default_rng(20260818)
    source = rng.standard_normal((1, 1, args.samples)).astype("float32")
    teacher = HubertTeacher()
    teacher.load_pretrained(args.pdparams_path)
    teacher.eval()
    with paddle.no_grad():
        paddle_out = teacher(paddle.to_tensor(source)).numpy()

    session = ort.InferenceSession(str(args.onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"source": source})[0]
    if paddle_out.shape != onnx_out.shape:
        raise ValueError(f"Output shape mismatch: Paddle={paddle_out.shape}, ONNX={onnx_out.shape}")
    diff = np.abs(paddle_out - onnx_out)
    result = {
        "shape": list(paddle_out.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "paddle_min": float(paddle_out.min()),
        "paddle_max": float(paddle_out.max()),
        "onnx_min": float(onnx_out.min()),
        "onnx_max": float(onnx_out.max()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
