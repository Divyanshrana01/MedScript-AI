# Single loader for domain config YAML files. All code should read domain
# settings through load_domain_config() instead of parsing YAML directly,
# so swapping domains only ever requires a new YAML file, not code changes.

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent


def load_domain_config(name: str = "medical") -> dict:
    """Load configs/domain/<name>.yaml and return it as a dict."""
    path = _CONFIG_DIR / "domain" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
