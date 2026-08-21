from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fair_hpo.data.preprocessing import (
    fit_transform_train,
    transform_data,
)
from fair_hpo.evaluation.cv import get_inner_cv
from fair_hpo.evaluation.metrics import evaluate_predictions
from fair_hpo.models.factory import build_model


def _get_positive_probability(model, X) -> np.ndarray | None:
    """Return positive-class probabilities when available."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None

    return np.asarray(probabilities[:, 1])


def evaluate_configuration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_train: pd.Series,
    model_name: str,
    params: dict[str, Any],
    random_state: int = 1000,
    inner_random_state: int = 42,
) -> dict[str, float]:
    """
    Evaluate one hyperparameter configuration using inner CV.

    Preprocessing is fitted independently inside each inner fold.
    The validation fold is therefore never used to fit preprocessing.

    The returned metrics are the mean across inner folds.
    """

    inner_splits = get_inner_cv(
        X_train=X_train,
        y_train=y_train,
        random_state=inner_random_state,
    )

    fold_results: list[dict[str, float]] = []

    for train_idx, validation_idx in inner_splits:
        X_inner_train = X_train.iloc[train_idx]
        X_inner_validation = X_train.iloc[validation_idx]

        y_inner_train = y_train.iloc[train_idx]
        y_inner_validation = y_train.iloc[validation_idx]

        sensitive_inner_validation = sensitive_train.iloc[
            validation_idx
        ]

        # Fit preprocessing ONLY on this inner-training fold.
        preprocessor, X_inner_train_processed = (
            fit_transform_train(X_inner_train)
        )

        X_inner_validation_processed = transform_data(
            preprocessor,
            X_inner_validation,
        )

        model = build_model(
            model_name=model_name,
            params=params,
            random_state=random_state,
        )

        model.fit(
            X_inner_train_processed,
            y_inner_train,
        )

        y_pred = model.predict(
            X_inner_validation_processed
        )

        y_prob = _get_positive_probability(
            model,
            X_inner_validation_processed,
        )

        metrics = evaluate_predictions(
            y_true=y_inner_validation,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive=sensitive_inner_validation,
        )

        fold_results.append(metrics)

    metric_names = fold_results[0].keys()

    return {
        metric_name: float(
            np.nanmean(
                [
                    fold[metric_name]
                    for fold in fold_results
                ]
            )
        )
        for metric_name in metric_names
    }
