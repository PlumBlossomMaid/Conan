"""Tests for utils/libritts_index.py and entry/build_probe_set.py.

Covered:
- ``speaker_of`` reads the leading field of a LibriTTS filename.
- ``source_paths`` returns the recorded group attribute in group-name order and
  raises when an older file lacks it.
- ``build_mapping`` prefers the authoritative ``source_path`` attribute and falls
  back to a validated replay of the preprocessing split when it is absent.
- ``list_audio_paths`` finds audio files and orders them by size.
- ``reconstruct_valid_files`` replays the split exactly (sampled, then size-sorted)
  and is reproducible for a fixed seed.
- ``validate_mapping`` accepts a consistent frame/size mapping and rejects a
  permuted one, and counts byte-size ties as order-ambiguous.
- ``group_frames`` returns the mel length of every group in name order.
- ``select_utterances`` (probe-set builder) spreads its picks and is stable.
- ``choose_speakers`` prefers richer speakers and breaks ties by id.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entry.build_probe_set import choose_speakers, select_utterances
from utils.libritts_index import (
    BYTES_PER_FRAME,
    WAV_HEADER_BYTES,
    build_mapping,
    group_frames,
    list_audio_paths,
    reconstruct_valid_files,
    source_paths,
    speaker_of,
    validate_mapping,
)


def test_speaker_of_reads_leading_field():
    assert speaker_of("1001_134708_000013_000000.wav") == "1001"
    assert speaker_of("100_121669_000001_000000.wav") == "100"


def test_list_audio_paths_sorts_by_size_and_ignores_other_files(tmp_path):
    for name, size in (("b.wav", 300), ("a.wav", 100), ("c.wav", 200)):
        (tmp_path / name).write_bytes(b"\0" * size)
    (tmp_path / "notes.txt").write_text("ignore me")

    paths = list_audio_paths(tmp_path)

    assert [p.name for p in paths] == ["a.wav", "c.wav", "b.wav"]
    assert [p.stat().st_size for p in paths] == [100, 200, 300]


def test_reconstruct_valid_files_is_reproducible_and_size_sorted(tmp_path):
    for index in range(12):
        (tmp_path / f"{index:03d}.wav").write_bytes(b"\0" * (100 + index * 7))
    paths = list_audio_paths(tmp_path)

    first = reconstruct_valid_files(paths, 5, 1234)
    second = reconstruct_valid_files(paths, 5, 1234)
    other_seed = reconstruct_valid_files(paths, 5, 7)

    assert len(first) == 5
    assert [p.name for p in first] == [p.name for p in second]
    assert [p.stat().st_size for p in first] == sorted(p.stat().st_size for p in first)
    assert [p.name for p in first] != [p.name for p in other_seed] or len(paths) == 5


def test_validate_mapping_accepts_consistent_and_rejects_permuted():
    frames = np.array([30, 40, 50, 60], dtype=np.int64)
    sizes = WAV_HEADER_BYTES + frames * BYTES_PER_FRAME

    good = validate_mapping(frames, sizes)
    assert good["frames_non_decreasing"] is True
    assert good["size_frame_correlation"] > 0.999
    assert good["residual_max_bytes"] == 0
    assert good["ambiguous_groups"] == 0

    bad = validate_mapping(frames[::-1], sizes)
    assert bad["frames_non_decreasing"] is False
    assert bad["size_frame_correlation"] < 0.999


def test_validate_mapping_counts_size_ties_as_ambiguous():
    frames = np.array([30, 30, 40], dtype=np.int64)
    sizes = WAV_HEADER_BYTES + frames * BYTES_PER_FRAME

    report = validate_mapping(frames, sizes)

    assert report["ambiguous_groups"] == 2


def test_group_frames_returns_mel_length_in_name_order(tmp_path):
    h5_path = tmp_path / "tiny.h5"
    with h5py.File(h5_path, "w") as handle:
        for name, length in (("00000001", 7), ("00000000", 5)):
            group = handle.create_group(name)
            group.create_dataset("mel", data=np.zeros((80, length), dtype=np.float32))

    assert group_frames(h5_path) == [5, 7]


def test_select_utterances_spreads_and_is_stable():
    names = [f"u{index:02d}" for index in range(10)]

    picked = select_utterances(names, 4)

    assert picked == select_utterances(names, 4)
    assert len(picked) == 4
    assert picked == sorted(picked)
    assert picked[0] == "u00" and picked[-1] == "u07"
    # fewer available than requested -> take them all
    assert select_utterances(names[:2], 5) == ["u00", "u01"]


def test_choose_speakers_prefers_richer_then_lower_id():
    by_speaker = {
        "200": ["a"] * 5,
        "100": ["a"] * 5,
        "300": ["a"] * 2,
        "400": ["a"] * 9,
    }

    chosen = choose_speakers(by_speaker, speakers=3, per_speaker=5)

    assert chosen == ["400", "100", "200"]


def _write_h5(h5_path, lengths, with_source_path=True, wavs_dir=None):
    with h5py.File(h5_path, "w") as handle:
        for index, length in enumerate(lengths):
            group = handle.create_group(f"{index:08d}")
            group.create_dataset("mel", data=np.zeros((80, length), dtype=np.float32))
            if with_source_path:
                group.attrs["source_path"] = str(Path(wavs_dir) / f"{index:03d}_1_0_0.wav")


def test_source_paths_returns_recorded_paths_in_group_order(tmp_path):
    h5_path = tmp_path / "with_attrs.h5"
    _write_h5(h5_path, [5, 7], wavs_dir="/audio")

    paths = source_paths(h5_path)

    assert [Path(p).name for p in paths] == ["000_1_0_0.wav", "001_1_0_0.wav"]
    assert all(p.startswith("/audio") for p in paths)


def test_source_paths_raises_when_the_attribute_is_missing(tmp_path):
    h5_path = tmp_path / "without_attrs.h5"
    _write_h5(h5_path, [5], with_source_path=False)

    with pytest.raises(KeyError, match="source_path"):
        source_paths(h5_path)


def test_build_mapping_prefers_source_path_and_needs_no_wavs_dir(tmp_path):
    h5_path = tmp_path / "with_attrs.h5"
    _write_h5(h5_path, [5, 7], wavs_dir="/audio")

    filenames, report = build_mapping(h5_path)

    assert report["method"] == "source_path"
    assert filenames == ["000_1_0_0.wav", "001_1_0_0.wav"]
    assert report["distinct_speakers"] == 2  # the helper varies the speaker field


def test_build_mapping_replays_the_split_when_the_attribute_is_absent(tmp_path):
    frames = [30, 40, 50]
    names = ["100_1_0_0.wav", "200_2_0_0.wav", "300_3_0_0.wav"]
    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir()
    for name, count in zip(names, frames):
        (wavs_dir / name).write_bytes(b"\0" * int(WAV_HEADER_BYTES + count * BYTES_PER_FRAME))
    h5_path = tmp_path / "without_attrs.h5"
    _write_h5(h5_path, frames, with_source_path=False)

    filenames, report = build_mapping(h5_path, wavs_dir=wavs_dir, n_valid=3, seed=1234)

    assert report["method"] == "replay"
    assert filenames == names  # already ordered by size, which is the write order
    assert report["frames_non_decreasing"] is True
    assert report["size_frame_correlation"] > 0.999
    assert report["ambiguous_groups"] == 0


def test_build_mapping_replay_requires_wavs_dir_and_n_valid(tmp_path):
    h5_path = tmp_path / "without_attrs.h5"
    _write_h5(h5_path, [30], with_source_path=False)

    with pytest.raises(ValueError, match="no source_path"):
        build_mapping(h5_path)


if __name__ == "__main__":
    import inspect
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                if "tmp_path" in inspect.signature(fn).parameters:
                    fn(tmp_path=tmp)
                else:
                    fn()
                print(f"✓ {name}")
    print("\nAll tests passed!")
