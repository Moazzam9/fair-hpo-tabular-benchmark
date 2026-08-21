from __future__ import annotations

from typing import Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a leakage-safe preprocessing transformer.

    Numeric columns:
        - median imputation

    Categorical columns:
        - most-frequent imputation
        - one-hot encoding

    IMPORTANT:
    The returned transformer must be fitted ONLY on training data.
    """

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_train(
    X_train: pd.DataFrame,
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """
    Fit preprocessing ONLY on training data and transform it.
    """

    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    feature_names = preprocessor.get_feature_names_out()

    X_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
        index=X_train.index,
    )

    return preprocessor, X_train_transformed


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform data using an already-fitted preprocessor.

    The transformer must NOT be fitted here.
    """

    X_transformed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()

    return pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )