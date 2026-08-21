import pytest
import pandas as pd

from fair_hpo.data.preprocessing import (
    fit_transform_train,
    transform_data,
)


DATASETS = [
    "wdbc",
    "adult",
    "bank_marketing",
]


@pytest.mark.parametrize("name", DATASETS)
def test_dataset(name):
    print(f"\n===== TESTING {name.upper()} =====")

    X = pd.read_csv(
        f"data/raw/{name}/features.csv"
    )

    # Simulate an outer train/test split.
    train_size = int(len(X) * 0.8)

    X_train = X.iloc[:train_size].copy()
    X_test = X.iloc[train_size:].copy()

    # Fit ONLY on training data.
    preprocessor, X_train_transformed = (
        fit_transform_train(X_train)
    )

    # Transform test using the already-fitted transformer.
    X_test_transformed = transform_data(
        preprocessor,
        X_test,
    )

    print(
        f"Original train shape: {X_train.shape}"
    )

    print(
        f"Original test shape:  {X_test.shape}"
    )

    print(
        f"Transformed train shape: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"Transformed test shape:  "
        f"{X_test_transformed.shape}"
    )

    # Basic checks.
    assert len(X_train_transformed) == len(X_train)
    assert len(X_test_transformed) == len(X_test)

    assert (
        X_train_transformed.shape[1]
        == X_test_transformed.shape[1]
    )

    assert X_train_transformed.isna().sum().sum() == 0
    assert X_test_transformed.isna().sum().sum() == 0

    print(f"{name}: PREPROCESSING PASS")