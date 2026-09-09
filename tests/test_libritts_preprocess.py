from pathlib import Path

import pytest

from entry.preprocess_libritts import LibriTTSPreprocessor


def test_preprocessor_validates_external_flat_audio_directory(tmp_path: Path):
    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()
    (audio_dir / "speaker_chapter_0001.wav").write_bytes(b"wav")
    (audio_dir / "metadata.txt").write_text("metadata", encoding="utf-8")
    (audio_dir / "nested").mkdir()
    (audio_dir / "nested" / "ignored.wav").write_bytes(b"nested")

    count, total_size = LibriTTSPreprocessor({"data": {"audio_dir": str(audio_dir)}}).run()

    assert count == 1
    assert total_size == 3


def test_preprocessor_rejects_missing_prepared_audio(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Prepared audio directory not found"):
        LibriTTSPreprocessor({"data": {"audio_dir": str(tmp_path / "missing")}}).run()


def test_preprocessor_rejects_empty_prepared_audio(tmp_path: Path):
    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()

    with pytest.raises(RuntimeError, match="No prepared audio files found"):
        LibriTTSPreprocessor({"data": {"audio_dir": str(audio_dir)}}).run()
