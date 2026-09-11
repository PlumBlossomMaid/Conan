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

from utils.config_utils import apply_overrides, load_config
from utils.dotdict import DotDict


def get_preprocess_class(preprocess_cls: str):
    module_path, _, class_name = preprocess_cls.rpartition(".")
    if not module_path:
        raise ValueError(f"preprocess_cls must be a full dotted path, got: {preprocess_cls!r}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Run a Conan preprocessing stage.")
    parser.add_argument("-c", "--config", required=True, help="Path to preprocessing config YAML")
    parser.add_argument(
        "-o", "--override", action="append", default=None,
        metavar="KEY=VALUE",
        help="Override a config value (repeatable, YAML-typed). "
             "Example: -o data.wavs_dir=/path/to/wavs -o preprocessing.device=iluvatar_gpu:0",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    apply_overrides(config, args.override)
    if "preprocess_cls" not in config:
        raise ValueError(f"{args.config} must define 'preprocess_cls'")
    DotDict(config).print_dict()

    preprocess_cls = get_preprocess_class(config["preprocess_cls"])
    preprocessor = preprocess_cls(config)
    preprocessor.run()


if __name__ == "__main__":
    main()
