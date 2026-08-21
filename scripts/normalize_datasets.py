from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


def normalize_wdbc() -> None:
    path = DATA_DIR / "wdbc"

    target_path = path / "target.csv"

    y = pd.read_csv(target_path)

    if "Diagnosis" not in y.columns:
        raise ValueError("WDBC target column 'Diagnosis' not found.")

    y["Diagnosis"] = y["Diagnosis"].astype(str).str.strip()

    invalid = set(y["Diagnosis"].unique()) - {"B", "M"}

    if invalid:
        raise ValueError(f"WDBC contains unexpected targets: {invalid}")

    y.to_csv(target_path, index=False)

    print("WDBC normalization: PASS")


def normalize_adult() -> None:
    path = DATA_DIR / "adult"

    target_path = path / "target.csv"

    y = pd.read_csv(target_path)

    if "income" not in y.columns:
        raise ValueError("Adult target column 'income' not found.")

    y["income"] = (
        y["income"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
    )

    invalid = set(y["income"].unique()) - {
        "<=50K",
        ">50K",
    }

    if invalid:
        raise ValueError(f"Adult contains unexpected targets: {invalid}")

    y.to_csv(target_path, index=False)

    print("Adult normalization: PASS")

    print("\nAdult target distribution:")
    print(y["income"].value_counts())


def normalize_bank() -> None:
    path = DATA_DIR / "bank_marketing"

    target_path = path / "target.csv"

    y = pd.read_csv(target_path)

    if "y" not in y.columns:
        raise ValueError("Bank target column 'y' not found.")

    y["y"] = y["y"].astype(str).str.strip()

    invalid = set(y["y"].unique()) - {
        "yes",
        "no",
    }

    if invalid:
        raise ValueError(f"Bank contains unexpected targets: {invalid}")

    y.to_csv(target_path, index=False)

    print("Bank Marketing normalization: PASS")


def main() -> None:
    normalize_wdbc()
    normalize_adult()
    normalize_bank()

    print("\nAll dataset normalization checks passed.")


if __name__ == "__main__":
    main()