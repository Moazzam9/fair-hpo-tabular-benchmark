from __future__ import annotations

from typing import Any

from sklearn.model_selection import ParameterSampler


def generate_random_configs(
    search_space: dict[str, Any],
    n_iter: int,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate hyperparameter configurations using random search.

    The function only generates configurations.
    Model fitting and evaluation are handled elsewhere.
    """

    if n_iter <= 0:
        raise ValueError("n_iter must be greater than zero.")

    sampler = ParameterSampler(
        param_distributions=search_space,
        n_iter=n_iter,
        random_state=random_state,
    )

    return list(sampler)