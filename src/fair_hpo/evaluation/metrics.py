from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def _as_numpy(values: Any) -> np.ndarray:
    """Convert pandas/numpy/list values to a 1-D numpy array."""
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.to_numpy()

    values = np.asarray(values)

    if values.ndim != 1:
        values = values.ravel()

    return values


def _get_binary_labels(
    y_true: Any,
    y_pred: Any,
) -> tuple[Any, Any]:
    """
    Return (negative_label, positive_label).

    The positive class is selected as follows:

    1. If label 1 exists, use 1.
    2. Otherwise, if a common positive-looking label exists, use it.
    3. Otherwise use the second sorted class.

    This supports both numeric targets such as 0/1 and string
    targets such as '<=50K' / '>50K'.
    """
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)

    labels = list(
        pd.unique(
            np.concatenate(
                [
                    y_true,
                    y_pred,
                ]
            )
        )
    )

    if len(labels) != 2:
        raise ValueError(
            "Binary classification metrics require exactly two "
            f"classes, but found: {labels}"
        )

    # Standard numeric binary classification.
    if 1 in labels:
        positive = 1
        negative = next(label for label in labels if label != 1)
        return negative, positive

    # Common Adult-income representation.
    for candidate in (
        ">50K",
        ">50K.",
        "yes",
        "Yes",
        "YES",
        "true",
        "True",
        "TRUE",
        "positive",
        "Positive",
        "POSITIVE",
        1,
    ):
        if candidate in labels:
            positive = candidate
            negative = next(
                label for label in labels if label != positive
            )
            return negative, positive

    # Fall back to deterministic ordering.
    try:
        sorted_labels = sorted(labels)
    except TypeError:
        sorted_labels = labels

    return sorted_labels[0], sorted_labels[1]


def _get_positive_probability(
    y_true: Any,
    y_prob: Any,
) -> np.ndarray | None:
    """
    Return probabilities for the positive class.

    y_prob is expected to contain probabilities for the two classes,
    normally in sklearn's class order.
    """
    if y_prob is None:
        return None

    probabilities = _as_numpy(y_prob).astype(float)

    if probabilities.ndim != 1:
        probabilities = probabilities.ravel()

    if len(probabilities) != len(_as_numpy(y_true)):
        raise ValueError(
            "y_prob must have the same number of observations as y_true."
        )

    return probabilities


def binary_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any = None,
) -> dict[str, float]:
    """
    Calculate binary classification metrics.

    Supports both numeric labels (0/1) and string labels
    such as '<=50K' / '>50K'.

    The positive class is detected automatically.
    """
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    negative_label, positive_label = _get_binary_labels(
        y_true,
        y_pred,
    )

    result: dict[str, float] = {}

    result["accuracy"] = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    result["balanced_accuracy"] = float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    result["f1"] = float(
        f1_score(
            y_true,
            y_pred,
            average="binary",
            pos_label=positive_label,
            zero_division=0,
        )
    )

    probabilities = _get_positive_probability(
        y_true,
        y_prob,
    )

    if probabilities is not None:
        # The runner supplies the probability for the positive class.
        y_binary = (
            y_true == positive_label
        ).astype(int)

        try:
            result["roc_auc"] = float(
                roc_auc_score(
                    y_binary,
                    probabilities,
                )
            )
        except ValueError:
            result["roc_auc"] = float("nan")
    else:
        result["roc_auc"] = float("nan")

    return result


def _fairness_groups(
    sensitive: Any,
) -> np.ndarray:
    """Convert sensitive attribute to a clean 1-D array."""
    sensitive = _as_numpy(sensitive)

    if len(sensitive) == 0:
        raise ValueError(
            "Sensitive attribute cannot be empty."
        )

    return sensitive


def demographic_parity_difference(
    y_pred: Any,
    sensitive: Any,
    positive_label: Any = 1,
) -> float:
    """
    Calculate demographic parity difference.

    Difference is:

        P(predicted positive | group 1)
        -
        P(predicted positive | group 0)

    For more than two groups, the returned value is the maximum
    difference between any two groups.
    """
    y_pred = _as_numpy(y_pred)
    sensitive = _fairness_groups(sensitive)

    if len(y_pred) != len(sensitive):
        raise ValueError(
            "y_pred and sensitive must have the same length."
        )

    groups = list(pd.unique(sensitive))

    rates: list[float] = []

    for group in groups:
        mask = sensitive == group

        if mask.sum() == 0:
            continue

        rate = float(
            np.mean(
                y_pred[mask] == positive_label
            )
        )

        rates.append(rate)

    if len(rates) < 2:
        return 0.0

    return float(
        max(rates) - min(rates)
    )


def equal_opportunity_difference(
    y_true: Any,
    y_pred: Any,
    sensitive: Any,
    positive_label: Any = 1,
) -> float:
    """
    Calculate equal opportunity difference.

    For each sensitive group:

        TPR = P(predicted positive | actual positive)

    The metric is the maximum difference between group TPRs.
    """
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    sensitive = _fairness_groups(sensitive)

    if not (
        len(y_true)
        == len(y_pred)
        == len(sensitive)
    ):
        raise ValueError(
            "y_true, y_pred, and sensitive must have "
            "the same length."
        )

    groups = list(pd.unique(sensitive))

    tprs: list[float] = []

    for group in groups:
        mask = sensitive == group

        actual_positive = (
            y_true[mask] == positive_label
        )

        denominator = int(
            actual_positive.sum()
        )

        if denominator == 0:
            continue

        true_positive = np.sum(
            (y_pred[mask] == positive_label)
            & actual_positive
        )

        tpr = float(
            true_positive / denominator
        )

        tprs.append(tpr)

    if len(tprs) < 2:
        return 0.0

    return float(
        max(tprs) - min(tprs)
    )


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    y_prob: Any = None,
    sensitive: Any = None,
) -> dict[str, float]:
    """
    Calculate classification and fairness metrics.

    This function is deliberately label-agnostic: targets may be
    numeric or strings.
    """
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)

    negative_label, positive_label = _get_binary_labels(
        y_true,
        y_pred,
    )

    results = binary_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    if sensitive is None:
        sensitive = np.array(
            ["all"] * len(y_true),
            dtype=object,
        )

    sensitive = _fairness_groups(sensitive)

    results[
        "demographic_parity_difference"
    ] = demographic_parity_difference(
        y_pred=y_pred,
        sensitive=sensitive,
        positive_label=positive_label,
    )

    results[
        "equal_opportunity_difference"
    ] = equal_opportunity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        positive_label=positive_label,
    )

    return results