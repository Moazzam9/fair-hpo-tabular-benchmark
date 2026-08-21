from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def _to_numpy(values: Any) -> np.ndarray:
    """Convert pandas/numpy/list-like values to a 1-D numpy array."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()

    return np.asarray(values).reshape(-1)


def binary_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
) -> dict[str, float]:
    """
    Calculate standard binary-classification metrics.

    Parameters
    ----------
    y_true:
        True binary labels.

    y_pred:
        Predicted binary labels.

    y_prob:
        Probability of the positive class.
        Required for ROC-AUC.

    Returns
    -------
    dict[str, float]
        Accuracy, balanced accuracy, F1 and ROC-AUC.
    """

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    results = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }

    if y_prob is not None:
        y_prob = _to_numpy(y_prob)

        try:
            results["roc_auc"] = float(
                roc_auc_score(y_true, y_prob)
            )
        except ValueError:
            results["roc_auc"] = float("nan")
    else:
        results["roc_auc"] = float("nan")

    return results


def demographic_parity_difference(
    y_pred: Any,
    sensitive: Any,
) -> float:
    """
    Calculate demographic parity difference.

    Difference between the highest and lowest positive
    prediction rates across sensitive groups.

    Lower absolute values indicate smaller disparity.
    """

    y_pred = _to_numpy(y_pred)
    sensitive = _to_numpy(sensitive)

    rates = []

    for group in np.unique(sensitive):
        mask = sensitive == group

        if mask.sum() == 0:
            continue

        rate = np.mean(y_pred[mask] == 1)
        rates.append(float(rate))

    if len(rates) < 2:
        return 0.0

    return float(max(rates) - min(rates))


def equal_opportunity_difference(
    y_true: Any,
    y_pred: Any,
    sensitive: Any,
) -> float:
    """
    Calculate equal opportunity difference.

    Difference between the highest and lowest true-positive
    rates across sensitive groups.

    Lower absolute values indicate smaller disparity.
    """

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    sensitive = _to_numpy(sensitive)

    tpr_values = []

    for group in np.unique(sensitive):
        mask = sensitive == group

        positives = y_true[mask] == 1

        if positives.sum() == 0:
            continue

        tpr = np.mean(
            y_pred[mask][positives] == 1
        )

        tpr_values.append(float(tpr))

    if len(tpr_values) < 2:
        return 0.0

    return float(
        max(tpr_values) - min(tpr_values)
    )


def fairness_metrics(
    y_true: Any,
    y_pred: Any,
    sensitive: Any,
) -> dict[str, float]:
    """
    Calculate the fairness metrics used by the benchmark.
    """

    return {
        "demographic_parity_difference":
            demographic_parity_difference(
                y_pred,
                sensitive,
            ),

        "equal_opportunity_difference":
            equal_opportunity_difference(
                y_true,
                y_pred,
                sensitive,
            ),
    }


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None,
    sensitive: Any,
) -> dict[str, float]:
    """
    Calculate predictive-performance and fairness metrics
    for one evaluation set.
    """

    results = binary_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    results.update(
        fairness_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive,
        )
    )

    return results