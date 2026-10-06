from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def require(config, *keys):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("missing configuration field: " + ".".join(keys))
        value = value[key]
    return value
