from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent


def load_domain_config(name: str = "medical") -> dict:
    path = _CONFIG_DIR / "domain" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
