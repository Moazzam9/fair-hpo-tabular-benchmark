from __future__ import annotations

from typing import Any


def default_random_forest_params() -> dict[str, Any]:
    """
    Baseline Random Forest configuration.

    These are infrastructure defaults only.
    They are NOT claimed to be the publication's
    final experimental settings.
    """

    return {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }


def default_xgboost_params() -> dict[str, Any]:
    """
    Baseline XGBoost configuration.

    These are infrastructure defaults only.
    """

    return {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }