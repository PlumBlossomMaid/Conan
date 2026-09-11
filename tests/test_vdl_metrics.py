"""Regression tests for the VisualDL metrics reader."""

import struct
from pathlib import Path

from visualdl.proto import record_pb2


def _make_vdl_file(path: Path, tags: dict) -> None:
    """Write a fake vdlrecords file: 8-byte header + Record payload each."""
    with open(path, "wb") as f:
        for tag, (step, value) in tags.items():
            rec = record_pb2.Record()
            v = rec.values.add()
            v.tag = tag
            v.id = step
            v.value = value
            payload = rec.SerializeToString()
            f.write(struct.pack("<I", len(payload)))
            f.write(struct.pack("<I", 0))
            f.write(payload)


def test_parse_vdl_file_reads_scalars(tmp_path):
    from utils.vdl_metrics import parse_vdl_file

    f = tmp_path / "vdlrecords.1.log"
    _make_vdl_file(f, {"train/loss": (100, 0.5), "val/loss": (500, 0.42)})
    curves = parse_vdl_file(str(f))
    assert curves["train/loss"] == [(100, 0.5)]
    assert curves["val/loss"][0][0] == 500
    assert abs(curves["val/loss"][0][1] - 0.42) < 1e-6


def test_parse_vdl_dir_merges_recursively_and_sorts(tmp_path):
    from utils.vdl_metrics import parse_vdl_dir

    version = tmp_path / "version_0"
    version.mkdir()
    _make_vdl_file(version / "vdlrecords.1.log", {"train/loss": (300, 0.4)})
    _make_vdl_file(version / "vdlrecords.2.log", {"train/loss": (100, 0.6)})
    curves = parse_vdl_dir(str(tmp_path))
    assert [p[0] for p in curves["train/loss"]] == [100, 300]
    assert abs(curves["train/loss"][0][1] - 0.6) < 1e-6
    assert abs(curves["train/loss"][1][1] - 0.4) < 1e-6


def test_parse_vdl_file_tolerates_trailing_partial_write(tmp_path):
    from utils.vdl_metrics import parse_vdl_file

    f = tmp_path / "vdlrecords.3.log"
    _make_vdl_file(f, {"train/loss": (0, 1.0)})
    with open(f, "ab") as fh:
        fh.write(b"\x10\x00")  # truncated header, no payload
    curves = parse_vdl_file(str(f))
    assert curves["train/loss"] == [(0, 1.0)]
