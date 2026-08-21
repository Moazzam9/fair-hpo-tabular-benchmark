from __future__ import annotations

from typing import Any, Callable

from skopt import Optimizer
from skopt.space import Categorical, Integer, Real


def build_skopt_space(
    search_space: dict[str, dict[str, Any]],
):
    """
    Convert our configuration representation into
    scikit-optimize dimensions.

    Example:

    {
        "n_estimators": {
            "type": "integer",
            "low": 50,
            "high": 300,
        },
        "max_depth": {
            "type": "integer",
            "low": 3,
            "high": 20,
        },
    }
    """

    dimensions = []

    for name, spec in search_space.items():
        parameter_type = spec["type"]

        if parameter_type == "integer":
            dimensions.append(
                Integer(
                    spec["low"],
                    spec["high"],
                    name=name,
                )
            )

        elif parameter_type == "real":
            dimensions.append(
                Real(
                    spec["low"],
                    spec["high"],
                    name=name,
                    prior=spec.get("prior", "uniform"),
                )
            )

        elif parameter_type == "categorical":
            dimensions.append(
                Categorical(
                    spec["values"],
                    name=name,
                )
            )

        else:
            raise ValueError(
                f"Unknown parameter type: {parameter_type}"
            )

    return dimensions


def create_bayesian_optimizer(
    search_space: dict[str, dict[str, Any]],
    random_state: int = 42,
) -> Optimizer:
    """
    Create a scikit-optimize Bayesian optimizer.

    This object proposes configurations.
    Actual model evaluation is handled by the experiment runner.
    """

    dimensions = build_skopt_space(search_space)

    return Optimizer(
        dimensions=dimensions,
        random_state=random_state,
    )


def ask_configuration(
    optimizer: Optimizer,
) -> dict[str, Any]:
    """
    Ask Bayesian optimizer for one configuration.
    """

    values = optimizer.ask()

    return {
        dimension.name: value
        for dimension, value in zip(
            optimizer.space.dimensions,
            values,
        )
    }


def tell_result(
    optimizer: Optimizer,
    configuration: dict[str, Any],
    score: float,
) -> None:
    """
    Report an evaluated configuration back to the optimizer.

    scikit-optimize minimizes its objective, therefore
    maximizing a metric requires passing -score.
    """

    values = [
        configuration[dimension.name]
        for dimension in optimizer.space.dimensions
    ]

    optimizer.tell(values, -score)