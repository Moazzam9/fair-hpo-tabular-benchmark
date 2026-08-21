from __future__ import annotations

from typing import Any

import pandas as pd

from fair_hpo.config.loader import load_fairness_config
from fair_hpo.evaluation.cv import get_outer_data


def get_sensitive_attribute(
    dataset_name: str,
    X: pd.DataFrame,
) -> pd.Series:
    """
    Return the configured sensitive attribute for a dataset.

    Raises an error when fairness evaluation is enabled but the
    configured column does not exist.
    """

    config = load_fairness_config()

    dataset_config = config.get("datasets", {}).get(
        dataset_name
    )

    if dataset_config is None:
        raise ValueError(
            f"No fairness configuration found for "
            f"dataset '{dataset_name}'."
        )

    enabled = bool(dataset_config.get("enabled", False))

    if not enabled:
        raise ValueError(
            f"Fairness evaluation is disabled for "
            f"dataset '{dataset_name}'."
        )

    sensitive_name = dataset_config.get(
        "sensitive_attribute"
    )

    if not sensitive_name:
        raise ValueError(
            f"No sensitive_attribute configured for "
            f"dataset '{dataset_name}'."
        )

    if sensitive_name not in X.columns:
        raise ValueError(
            f"Sensitive attribute '{sensitive_name}' "
            f"not found in dataset '{dataset_name}'."
        )

    return X[sensitive_name].copy()


def get_fairness_groups(
    dataset_name: str,
) -> dict[str, Any]:
    """
    Return configured fairness-group metadata.
    """

    config = load_fairness_config()

    dataset_config = config.get("datasets", {}).get(
        dataset_name
    )

    if dataset_config is None:
        raise ValueError(
            f"No fairness configuration found for "
            f"dataset '{dataset_name}'."
        )

    return {
        "sensitive_attribute": dataset_config.get(
            "sensitive_attribute"
        ),
        "privileged_group": dataset_config.get(
            "privileged_group"
        ),
        "unprivileged_group": dataset_config.get(
            "unprivileged_group"
        ),
        "positive_label": dataset_config.get(
            "positive_label"
        ),
    }


def get_outer_data_with_sensitive(
    dataset_name: str,
    fold_id: int,
):
    """
    Load an outer fold together with its sensitive attribute.

    The sensitive attribute is returned separately so the experiment
    runner can decide explicitly whether it is included as a model
    feature.
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = get_outer_data(
        dataset_name,
        fold_id,
    )

    sensitive_train = get_sensitive_attribute(
        dataset_name,
        X_train,
    )

    sensitive_test = get_sensitive_attribute(
        dataset_name,
        X_test,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_train,
        sensitive_test,
    )
