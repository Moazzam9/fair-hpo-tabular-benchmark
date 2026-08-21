from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    params: dict[str, Any] | None = None,
    random_state: int = 1000,
) -> RandomForestClassifier:
    """
    Build a Random Forest classifier.

    The model is deliberately single-threaded so that runtime
    comparisons between optimizers are not confounded by
    different levels of model parallelism.
    """

    params = dict(params or {})

    params.setdefault("n_jobs", 1)
    params.setdefault("random_state", random_state)

    return RandomForestClassifier(**params)


def build_xgboost(
    params: dict[str, Any] | None = None,
    random_state: int = 1000,
):
    """
    Build an XGBoost classifier.

    XGBoost is imported lazily so that the rest of the project
    can still be imported if xgboost has not yet been installed.
    """

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. "
            "Install it with: pip install xgboost"
        ) from exc

    params = dict(params or {})

    params.setdefault("n_jobs", 1)
    params.setdefault("random_state", random_state)
    params.setdefault("eval_metric", "logloss")

    return XGBClassifier(**params)


def build_model(
    model_name: str,
    params: dict[str, Any] | None = None,
    random_state: int = 1000,
):
    """
    General model factory.
    """

    if model_name == "random_forest":
        return build_random_forest(
            params=params,
            random_state=random_state,
        )

    if model_name == "xgboost":
        return build_xgboost(
            params=params,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )