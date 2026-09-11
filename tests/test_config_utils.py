"""Shared config loader and CLI override helpers."""

import pytest

from utils.config_utils import apply_overrides, load_config


def test_apply_overrides_sets_nested_and_typed_values():
    config = {"data": {"batch_size": 4, "keep": 1}, "training": {"steps": 100}}

    apply_overrides(config, ["data.batch_size=16", "training.steps=1", "new.key=0.5"])

    assert config["data"]["batch_size"] == 16
    assert config["data"]["keep"] == 1
    assert config["training"]["steps"] == 1
    assert config["new"]["key"] == 0.5


def test_apply_overrides_rejects_missing_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["badkey"])


def test_apply_overrides_ignores_none():
    config = {"a": 1}
    assert apply_overrides(config, None) is config
    assert config == {"a": 1}


def test_load_config_merges_base_config(tmp_path):
    base = tmp_path / "base.yaml"
    stage = tmp_path / "stage.yaml"
    base.write_text("data:\n  a: 1\n  b: 2\nseed: 7\n", encoding="utf-8")
    stage.write_text(f"base_config: {base}\nseed: 42\n", encoding="utf-8")

    config = load_config(str(stage))

    assert config["data"] == {"a": 1, "b": 2}
    assert config["seed"] == 42
