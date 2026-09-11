"""Parse VisualDL record files into plain dicts of tag -> [(step, value)].

The logger writes protobuf ``Record`` messages into ``vdlrecords.*.log``.
Each record is ``<4-byte LE length><4-byte zero><payload>``. This module
decodes them so training curves (train/loss, val/loss, perf/*) can be
reviewed from the shell without opening VDL.

Usage::

    from utils.vdl_metrics import parse_vdl_dir

    curves = parse_vdl_dir("logs/content_extractor_ce/ocean_logs")
    print(curves["val/loss"])        # [(step, value), ...]
"""

import struct
from pathlib import Path
from typing import Dict, List, Tuple

from visualdl.proto import record_pb2

Tag = str
Point = Tuple[int, float]
Curves = Dict[Tag, List[Point]]


def parse_vdl_file(path: str) -> Curves:
    """Parse a single vdlrecords file into tag -> [(step, value), ...]."""
    curves: Curves = {}
    data = Path(path).read_bytes()
    pos = 0
    while pos + 8 <= len(data):
        (length,) = struct.unpack("<I", data[pos : pos + 4])
        payload = data[pos + 8 : pos + 8 + length]
        try:
            rec = record_pb2.Record()
            rec.ParseFromString(payload)
        except Exception:
            break  # trailing partial write
        for v in rec.values:
            if v.tag and v.HasField("value"):
                curves.setdefault(v.tag, []).append((v.id, float(v.value)))
        pos += 8 + length
    return curves


def parse_vdl_dir(log_dir: str) -> Curves:
    """Parse every ``vdlrecords*`` file under ``log_dir`` and merge by tag.

    Searches recursively (runs write under ``<log_dir>/<version>``), and
    records from different files are appended in file order; callers that
    need strict step ordering should sort the merged points themselves.
    """
    merged: Curves = {}
    for path in sorted(Path(log_dir).rglob("vdlrecords*")):
        for tag, points in parse_vdl_file(str(path)).items():
            merged.setdefault(tag, []).extend(points)
    for points in merged.values():
        points.sort(key=lambda p: p[0])
    return merged
