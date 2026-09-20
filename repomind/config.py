import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    # Resolve ${VAR} placeholders from .env
    def resolve(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.getenv(value[2:-1], "")
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(raw)
