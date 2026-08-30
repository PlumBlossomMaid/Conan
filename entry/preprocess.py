"""Unified preprocessing entry point for Conan.

The preprocessing task is selected by ``preprocess_cls`` in the config file.

Usage:
    python entry/preprocess.py -c configs/preprocess_hubert.yaml
"""

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from utils.dotdict import DotDict


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parents = config.pop("base_config", None)
    if not parents:
        return config
    if isinstance(parents, str):
        parents = [parents]

    merged = {}
    for parent in parents:
        merged = _deep_update(merged, load_config(parent))
    return _deep_update(merged, config)


def _deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def get_preprocess_class(preprocess_cls: str):
    module_path, _, class_name = preprocess_cls.rpartition(".")
    if not module_path:
        raise ValueError(f"preprocess_cls must be a full dotted path, got: {preprocess_cls!r}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Run a Conan preprocessing stage.")
    parser.add_argument("-c", "--config", required=True, help="Path to preprocessing config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    if "preprocess_cls" not in config:
        raise ValueError(f"{args.config} must define 'preprocess_cls'")
    DotDict(config).print_dict()

    preprocess_cls = get_preprocess_class(config["preprocess_cls"])
    preprocessor = preprocess_cls(config)
    preprocessor.run()


if __name__ == "__main__":
    main()
