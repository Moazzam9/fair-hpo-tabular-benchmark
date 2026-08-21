from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


DATA_DIR = Path("data/raw")
SPLIT_DIR = Path("data/splits")

DATASETS = [
    "wdbc",
    "adult",
    "bank_marketing",
]


def load_dataset(name: str):
    """Load normalized features and target."""

    X_path = DATA_DIR / name / "features.csv"
    y_path = DATA_DIR / name / "target.csv"

    if not X_path.exists():
        raise FileNotFoundError(X_path)

    if not y_path.exists():
        raise FileNotFoundError(y_path)

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0]

    return X, y


def make_outer_splits(
    X,
    y,
    n_splits=5,
    random_state=42,
):
    """Create fixed stratified outer CV splits."""

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    return list(splitter.split(X, y))


def save_outer_splits(name: str, n_splits=5, random_state=42):
    """Create and save outer CV indices for one dataset."""

    X, y = load_dataset(name)

    splits = make_outer_splits(
        X,
        y,
        n_splits=n_splits,
        random_state=random_state,
    )

    output_dir = SPLIT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):

        train_path = output_dir / f"outer_fold_{fold_id}_train.csv"
        test_path = output_dir / f"outer_fold_{fold_id}_test.csv"

        pd.DataFrame({"index": train_idx}).to_csv(
            train_path,
            index=False,
        )

        pd.DataFrame({"index": test_idx}).to_csv(
            test_path,
            index=False,
        )

        print(
            f"{name}: outer fold {fold_id} "
            f"train={len(train_idx)} "
            f"test={len(test_idx)}"
        )


def make_all_splits():
    """Generate fixed outer splits for all datasets."""

    for name in DATASETS:
        print(f"\nGenerating splits for {name}...")
        save_outer_splits(
            name=name,
            n_splits=5,
            random_state=42,
        )

    print("\nAll outer splits generated successfully.")


if __name__ == "__main__":
    make_all_splits()