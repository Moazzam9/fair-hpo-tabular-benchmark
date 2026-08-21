from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    """

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}"
        )

    return data


def load_dataset_config() -> dict[str, Any]:
    return load_yaml("configs/datasets.yaml")


def load_fairness_config() -> dict[str, Any]:
    return load_yaml("configs/fairness.yaml")