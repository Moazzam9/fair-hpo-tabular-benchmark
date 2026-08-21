from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


EXPECTED = {
    "wdbc": {
        "rows": 569,
        "features": 30,
    },
    "adult": {
        "rows": 48842,
        "features": 14,
    },
    "bank_marketing": {
        "rows": 45211,
        "features": 16,
    },
}


def validate_dataset(name: str):
    features_path = DATA_DIR / name / "features.csv"
    target_path = DATA_DIR / name / "target.csv"

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    if not target_path.exists():
        raise FileNotFoundError(target_path)

    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)

    expected = EXPECTED[name]

    assert len(X) == expected["rows"], (
        f"{name}: expected {expected['rows']} rows, "
        f"got {len(X)}"
    )

    assert X.shape[1] == expected["features"], (
        f"{name}: expected {expected['features']} features, "
        f"got {X.shape[1]}"
    )

    assert len(X) == len(y), (
        f"{name}: feature/target row mismatch"
    )

    print(f"{name}: PASS")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")


def validate_all():
    for name in EXPECTED:
        validate_dataset(name)


if __name__ == "__main__":
    validate_all()