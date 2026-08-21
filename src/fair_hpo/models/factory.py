from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class XGBClassifierWrapper:
    """
    Wrapper around XGBClassifier that supports arbitrary binary
    class labels such as '<=50K' and '>50K'.

    The underlying XGBoost model receives 0/1 labels, while
    predict() and predict_proba() expose the original labels/
    probability convention expected by the rest of the project.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        random_state: int = 1000,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. "
                "Install it with: pip install xgboost"
            ) from exc

        self._params = dict(params or {})

        self._params.setdefault("n_jobs", 1)
        self._params.setdefault(
            "random_state",
            random_state,
        )
        self._params.setdefault(
            "eval_metric",
            "logloss",
        )

        self._model = XGBClassifier(
            **self._params
        )

        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        y_array = np.asarray(y)

        classes = np.unique(y_array)

        if len(classes) != 2:
            raise ValueError(
                "XGBoost wrapper requires exactly two "
                f"classes, got {classes.tolist()}"
            )

        self.classes_ = classes

        y_encoded = np.where(
            y_array == classes[0],
            0,
            1,
        )

        self._model.fit(
            X,
            y_encoded,
        )

        return self

    def predict(self, X):
        if self.classes_ is None:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        encoded_predictions = self._model.predict(X)

        encoded_predictions = np.asarray(
            encoded_predictions,
            dtype=int,
        )

        return self.classes_[encoded_predictions]

    def predict_proba(self, X):
        if self.classes_ is None:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        return self._model.predict_proba(X)

    def __getattr__(self, name):
        """
        Delegate attributes such as n_estimators,
        max_depth, learning_rate, etc. to XGBoost.
        """

        if name.startswith("_"):
            raise AttributeError(name)

        return getattr(
            self._model,
            name,
        )


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

    params.setdefault(
        "n_jobs",
        1,
    )

    params.setdefault(
        "random_state",
        random_state,
    )

    return RandomForestClassifier(
        **params
    )


def build_xgboost(
    params: dict[str, Any] | None = None,
    random_state: int = 1000,
):
    """
    Build an XGBoost classifier capable of handling string
    binary target labels.
    """

    return XGBClassifierWrapper(
        params=params,
        random_state=random_state,
    )


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