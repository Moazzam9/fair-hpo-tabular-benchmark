from __future__ import annotations

from typing import Any

import pandas as pd

from fair_hpo.experiments.runner import (
    evaluate_configuration_inner_cv,
    fit_final_outer_model,
)
from fair_hpo.optimizers.bayesian_search import (
    create_bayesian_optimizer,
    ask_configuration,
    tell_result,
)
from fair_hpo.optimizers.random_search import (
    generate_random_configs,
)


def objective_score(
    metrics: dict[str, float],
    objective: str = "balanced_accuracy",
    fairness_weight: float = 0.0,
) -> float:
    """
    Calculate the scalar objective used by HPO.

    Higher is better.

    fairness_weight controls the penalty applied to
    fairness disparity metrics.
    """

    if objective not in metrics:
        raise ValueError(
            f"Unknown objective metric: {objective}"
        )

    score = metrics[objective]

    if pd.isna(score):
        return float("-inf")

    if fairness_weight > 0:
        dpd = metrics.get(
            "mean_demographic_parity_difference",
            0.0,
        )

        eod = metrics.get(
            "mean_equal_opportunity_difference",
            0.0,
        )

        score -= fairness_weight * (
            abs(dpd) + abs(eod)
        )

    return float(score)


def run_random_search(
    dataset_name: str,
    fold_id: int,
    model_name: str,
    search_space: dict[str, Any],
    n_iter: int,
    sensitive_column: str | None = None,
    objective: str = "mean_balanced_accuracy",
    fairness_weight: float = 0.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run random HPO on one outer fold.
    """

    configurations = generate_random_configs(
        search_space=search_space,
        n_iter=n_iter,
        random_state=random_state,
    )

    history = []

    best_config = None
    best_score = float("-inf")
    best_metrics = None

    for iteration, params in enumerate(
        configurations,
        start=1,
    ):
        metrics = evaluate_configuration_inner_cv(
            dataset_name=dataset_name,
            fold_id=fold_id,
            model_name=model_name,
            params=params,
            sensitive_column=sensitive_column,
            random_state=1000,
            cv_random_state=42,
        )

        score = objective_score(
            metrics,
            objective=objective,
            fairness_weight=fairness_weight,
        )

        row = {
            "iteration": iteration,
            "optimizer": "random",
            **params,
            **metrics,
            "objective_score": score,
        }

        history.append(row)

        if score > best_score:
            best_score = score
            best_config = params
            best_metrics = metrics

    if best_config is None:
        raise RuntimeError(
            "Random search did not produce a valid configuration."
        )

    outer_result = fit_final_outer_model(
        dataset_name=dataset_name,
        fold_id=fold_id,
        model_name=model_name,
        params=best_config,
        sensitive_column=sensitive_column,
        random_state=1000,
    )

    return {
        "best_params": best_config,
        "best_inner_metrics": best_metrics,
        "best_objective_score": best_score,
        "outer_result": outer_result,
        "history": history,
    }


def run_bayesian_search(
    dataset_name: str,
    fold_id: int,
    model_name: str,
    search_space: dict[str, dict[str, Any]],
    n_iter: int,
    sensitive_column: str | None = None,
    objective: str = "mean_balanced_accuracy",
    fairness_weight: float = 0.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run Bayesian HPO on one outer fold.
    """

    optimizer = create_bayesian_optimizer(
        search_space=search_space,
        random_state=random_state,
    )

    history = []

    best_config = None
    best_score = float("-inf")
    best_metrics = None

    for iteration in range(1, n_iter + 1):
        params = ask_configuration(
            optimizer
        )

        metrics = evaluate_configuration_inner_cv(
            dataset_name=dataset_name,
            fold_id=fold_id,
            model_name=model_name,
            params=params,
            sensitive_column=sensitive_column,
            random_state=1000,
            cv_random_state=42,
        )

        score = objective_score(
            metrics,
            objective=objective,
            fairness_weight=fairness_weight,
        )

        tell_result(
            optimizer=optimizer,
            configuration=params,
            score=score,
        )

        row = {
            "iteration": iteration,
            "optimizer": "bayesian",
            **params,
            **metrics,
            "objective_score": score,
        }

        history.append(row)

        if score > best_score:
            best_score = score
            best_config = params
            best_metrics = metrics

    if best_config is None:
        raise RuntimeError(
            "Bayesian search did not produce a valid configuration."
        )

    outer_result = fit_final_outer_model(
        dataset_name=dataset_name,
        fold_id=fold_id,
        model_name=model_name,
        params=best_config,
        sensitive_column=sensitive_column,
        random_state=1000,
    )

    return {
        "best_params": best_config,
        "best_inner_metrics": best_metrics,
        "best_objective_score": best_score,
        "outer_result": outer_result,
        "history": history,
    }