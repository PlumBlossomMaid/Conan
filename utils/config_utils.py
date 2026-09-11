"""Config loading and CLI override helpers shared by entry points."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> dict:
    """Load a stage config, merging it over ``base_config`` parents if declared.

    Args:
        config_path: Path to the stage YAML (absolute or project-relative).

    Returns:
        Merged configuration dict.
    """
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
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def apply_overrides(config: dict, overrides) -> dict:
    """Apply ``KEY=VALUE`` CLI overrides onto a config.

    Values are YAML-parsed so numbers and booleans keep their types.
    Nested keys create intermediate dicts on demand.

    Args:
        config: Config dict to mutate in place.
        overrides: Iterable of ``key=value`` strings, or None.

    Returns:
        The same config dict, updated.
    """
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"override expects 'key=value', got: {override!r} "
                "(e.g. data.wavs_dir=/path/to/wavs)"
            )
        key, _, raw = override.partition("=")
        value = yaml.safe_load(raw)
        target = config
        for part in key.split(".")[:-1]:
            target = target.setdefault(part, {})
        target[key.split(".")[-1]] = value
    return config
