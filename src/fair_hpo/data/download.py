from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATA_DIR = Path("data/raw")


DATASETS = {
    "wdbc": {
        "uci_id": 17,
        "expected_rows": 569,
        "expected_features": 30,
    },
    "adult": {
        "uci_id": 2,
        "expected_rows": 48842,
        "expected_features": 14,
    },
    "bank_marketing": {
        "uci_id": 222,
        "expected_rows": 45211,
        "expected_features": 16,
    },
}


def download_dataset(name: str) -> None:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    config = DATASETS[name]

    print("=" * 70)
    print(f"Downloading: {name}")
    print(f"UCI dataset ID: {config['uci_id']}")
    print("=" * 70)

    dataset = fetch_ucirepo(id=config["uci_id"])

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    print(f"Downloaded X shape: {X.shape}")
    print(f"Downloaded y shape: {y.shape}")

    if len(X) != config["expected_rows"]:
        raise ValueError(
            f"{name}: expected {config['expected_rows']} rows, "
            f"got {len(X)}"
        )

    if X.shape[1] != config["expected_features"]:
        raise ValueError(
            f"{name}: expected {config['expected_features']} features, "
            f"got {X.shape[1]}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name}: feature/target row mismatch"
        )

    output_dir = DATA_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features.csv"
    target_path = output_dir / "target.csv"

    X.to_csv(features_path, index=False)
    y.to_csv(target_path, index=False)

    print()
    print(f"Saved features: {features_path}")
    print(f"Saved target:   {target_path}")
    print()
    print(f"{name}: DOWNLOAD PASS")


def download_all() -> None:
    for name in DATASETS:
        download_dataset(name)


if __name__ == "__main__":
    download_all()