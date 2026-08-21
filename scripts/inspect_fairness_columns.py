from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


DATASETS = [
    "wdbc",
    "adult",
    "bank_marketing",
]


def inspect_dataset(name: str):
    print()
    print("=" * 70)
    print(f"{name.upper()}")
    print("=" * 70)

    features_path = DATA_DIR / name / "features.csv"
    target_path = DATA_DIR / name / "target.csv"

    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)

    print("\nFEATURE COLUMNS:")
    for i, column in enumerate(X.columns):
        print(
            f"{i:>3}: {column:<30} "
            f"dtype={X[column].dtype}"
        )

    print("\nTARGET COLUMN:")
    print(y.columns.tolist())

    for column in X.columns:
        if X[column].dtype == "object":
            print()
            print(f"CATEGORICAL: {column}")
            print(X[column].value_counts(dropna=False).head(20))

    print()


def main():
    for dataset in DATASETS:
        inspect_dataset(dataset)


if __name__ == "__main__":
    main()