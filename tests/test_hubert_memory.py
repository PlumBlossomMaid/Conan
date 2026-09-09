"""Tests for memory guard and resumable HDF5 output helpers."""

import h5py
import pytest

from entry.preprocess_hubert import _completed_groups, _memory_budget_ratio


def test_memory_budget_auto_is_conservative():
    assert _memory_budget_ratio({"memory_limit": "auto"}) in (pytest.approx(0.65), pytest.approx(0.80))


def test_memory_budget_accepts_configured_ratio():
    assert _memory_budget_ratio({"memory_limit": 0.7}) == pytest.approx(0.7)


def test_memory_budget_rejects_unsafe_values():
    with pytest.raises(ValueError):
        _memory_budget_ratio({"memory_limit": 1})


def test_completed_groups_removes_incomplete_and_non_contiguous_groups(tmp_path):
    path = tmp_path / "features.h5"
    with h5py.File(path, "w") as h5f:
        for key in ("00000000", "00000001", "00000003"):
            group = h5f.create_group(key)
            if key != "00000001":
                group.create_dataset("mel", data=[1])
                group.create_dataset("hubert", data=[1])

    with h5py.File(path, "a") as h5f:
        assert _completed_groups(h5f) == 1
        assert list(h5f.keys()) == ["00000000"]
