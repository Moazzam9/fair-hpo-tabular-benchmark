from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fair_hpo.data.preprocessing import (
    fit_transform_train,
    transform_data,
)
from fair_hpo.evaluation.cv import (
    get_inner_cv,
    get_outer_data,
)
from fair_hpo.evaluation.metrics import evaluate_predictions
from fair_hpo.models.factory import build_model


def _is_xgboost(model_name: str) -> bool:
    """Return True when the selected model is XGBoost."""
    return model_name.lower() in {
        "xgboost",
        "xgb",
    }


def _encode_binary_target(
    y_train: pd.Series,
    y_validation: pd.Series | None = None,
    y_test: pd.Series | None = None,
):
    """
    Encode a binary target for models that require numeric class labels.

    The encoder is fitted ONLY on the training target.

    Example for Adult:
        <=50K -> 0
        >50K  -> 1

    The mapping is learned from y_train rather than hard-coded.
    """

    encoder = LabelEncoder()

    y_train_encoded = pd.Series(
        encoder.fit_transform(y_train),
        index=y_train.index,
        name=y_train.name,
    )

    if len(encoder.classes_) != 2:
        raise ValueError(
            "Binary classification requires exactly two target "
            f"classes, but found: {list(encoder.classes_)}"
        )

    y_validation_encoded = None

    if y_validation is not None:
        y_validation_encoded = pd.Series(
            encoder.transform(y_validation),
            index=y_validation.index,
            name=y_validation.name,
        )

    y_test_encoded = None

    if y_test is not None:
        y_test_encoded = pd.Series(
            encoder.transform(y_test),
            index=y_test.index,
            name=y_test.name,
        )

    return (
        encoder,
        y_train_encoded,
        y_validation_encoded,
        y_test_encoded,
    )


def _prepare_training_target(
    model_name: str,
    y_train: pd.Series,
    y_validation: pd.Series | None = None,
):
    """
    Prepare targets for model fitting.

    XGBoost receives numeric 0/1 labels.

    Other models receive the original labels.

    Returns:
        target_encoder
        model_training_target
        encoded_validation_target
    """

    if not _is_xgboost(model_name):
        return (
            None,
            y_train,
            y_validation,
        )

    (
        encoder,
        y_train_encoded,
        y_validation_encoded,
        _,
    ) = _encode_binary_target(
        y_train=y_train,
        y_validation=y_validation,
    )

    return (
        encoder,
        y_train_encoded,
        y_validation_encoded,
    )


def _restore_predictions(
    model_name: str,
    predictions,
    target_encoder: LabelEncoder | None,
):
    """
    Convert model predictions back to the original target labels.

    This ensures metric evaluation receives the same label representation
    as y_true.
    """

    if not _is_xgboost(model_name):
        return predictions

    if target_encoder is None:
        raise RuntimeError(
            "XGBoost predictions require a fitted target encoder."
        )

    predictions = pd.Series(predictions).astype(int).to_numpy()

    return target_encoder.inverse_transform(
        predictions
    )


def _get_prediction_probabilities(
    model,
    X,
):
    """
    Return probability of the encoded positive class.

    LabelEncoder maps the two classes to:
        class 0 -> negative
        class 1 -> positive

    Therefore probability column 1 is the probability of the positive
    class for both sklearn models trained on encoded targets and
    XGBoost.
    """

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        return None

    if probabilities.shape[1] != 2:
        return None

    return probabilities[:, 1]


def evaluate_configuration_inner_cv(
    dataset_name: str,
    fold_id: int,
    model_name: str,
    params: dict[str, Any],
    sensitive_column: str | None = None,
    random_state: int = 1000,
    cv_random_state: int = 42,
) -> dict[str, float]:
    """
    Evaluate one hyperparameter configuration using inner CV.

    IMPORTANT:
    Preprocessing is fitted independently inside every inner fold.

    Sensitive attributes are never supplied to the model.

    For XGBoost:
        target labels are encoded to 0/1 for training,
        then predictions are converted back to original labels
        before metric evaluation.
    """

    (
        X_outer_train,
        _,
        y_outer_train,
        _,
    ) = get_outer_data(
        dataset_name,
        fold_id,
    )

    # --------------------------------------------------
    # Remove sensitive attribute from model features.
    # --------------------------------------------------

    if sensitive_column is not None:
        if sensitive_column not in X_outer_train.columns:
            raise ValueError(
                f"Sensitive column '{sensitive_column}' "
                f"not found in dataset '{dataset_name}'."
            )

        sensitive = X_outer_train[
            sensitive_column
        ].copy()

        X_model = X_outer_train.drop(
            columns=[sensitive_column]
        ).copy()

    else:
        sensitive = None
        X_model = X_outer_train.copy()

    # --------------------------------------------------
    # Build inner CV splits.
    # --------------------------------------------------

    inner_splits = get_inner_cv(
        X_model,
        y_outer_train,
        random_state=cv_random_state,
    )

    fold_results: list[dict[str, float]] = []

    for inner_fold_id, (
        train_idx,
        validation_idx,
    ) in enumerate(
        inner_splits,
        start=1,
    ):
        # --------------------------------------------------
        # Split inner training/validation data.
        # --------------------------------------------------

        X_inner_train = (
            X_model.iloc[train_idx].copy()
        )

        X_inner_validation = (
            X_model.iloc[validation_idx].copy()
        )

        y_inner_train = (
            y_outer_train.iloc[train_idx].copy()
        )

        y_inner_validation = (
            y_outer_train.iloc[
                validation_idx
            ].copy()
        )

        # --------------------------------------------------
        # Sensitive attribute is NOT used as a model feature.
        # --------------------------------------------------

        if sensitive is not None:
            sensitive_validation = sensitive.iloc[
                validation_idx
            ].copy()
        else:
            sensitive_validation = None

        # --------------------------------------------------
        # Leakage-safe preprocessing.
        #
        # Fit ONLY on inner training data.
        # --------------------------------------------------

        (
            preprocessor,
            X_inner_train_processed,
        ) = fit_transform_train(
            X_inner_train
        )

        X_inner_validation_processed = (
            transform_data(
                preprocessor,
                X_inner_validation,
            )
        )

        # --------------------------------------------------
        # Build model.
        # --------------------------------------------------

        model = build_model(
            model_name=model_name,
            params=params,
            random_state=random_state,
        )

        # --------------------------------------------------
        # Prepare target.
        #
        # XGBoost requires numeric binary labels.
        # Random Forest and other sklearn models keep
        # their original labels.
        # --------------------------------------------------

        (
            target_encoder,
            y_train_model,
            _,
        ) = _prepare_training_target(
            model_name=model_name,
            y_train=y_inner_train,
            y_validation=y_inner_validation,
        )

        # --------------------------------------------------
        # Fit model.
        # --------------------------------------------------

        model.fit(
            X_inner_train_processed,
            y_train_model,
        )

        # --------------------------------------------------
        # Generate predictions.
        # --------------------------------------------------

        y_pred_model = model.predict(
            X_inner_validation_processed
        )

        y_pred = _restore_predictions(
            model_name=model_name,
            predictions=y_pred_model,
            target_encoder=target_encoder,
        )

        # --------------------------------------------------
        # Generate probabilities.
        # --------------------------------------------------

        y_prob = _get_prediction_probabilities(
            model=model,
            X=X_inner_validation_processed,
        )

        # --------------------------------------------------
        # Evaluate.
        #
        # IMPORTANT:
        # y_true and y_pred use the original target labels.
        # --------------------------------------------------

        if sensitive_validation is not None:
            metrics = evaluate_predictions(
                y_true=y_inner_validation,
                y_pred=y_pred,
                y_prob=y_prob,
                sensitive=sensitive_validation,
            )
        else:
            metrics = evaluate_predictions(
                y_true=y_inner_validation,
                y_pred=y_pred,
                y_prob=y_prob,
                sensitive=["all"]
                * len(y_inner_validation),
            )

        metrics["inner_fold"] = float(
            inner_fold_id
        )

        fold_results.append(metrics)

    # --------------------------------------------------
    # Aggregate inner-fold metrics.
    # --------------------------------------------------

    result: dict[str, float] = {}

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "demographic_parity_difference",
        "equal_opportunity_difference",
    ]

    for metric_name in metric_names:
        values = [
            row[metric_name]
            for row in fold_results
        ]

        result[
            f"mean_{metric_name}"
        ] = float(
            pd.Series(values).mean()
        )

    return result


def fit_final_outer_model(
    dataset_name: str,
    fold_id: int,
    model_name: str,
    params: dict[str, Any],
    sensitive_column: str | None = None,
    random_state: int = 1000,
) -> dict[str, Any]:
    """
    Fit the selected configuration on the complete outer-training set
    and evaluate once on the untouched outer test set.

    For XGBoost:
        target labels are encoded as 0/1 for training,
        predictions are converted back to original labels,
        and metrics are evaluated using original labels.
    """

    (
        X_outer_train,
        X_outer_test,
        y_outer_train,
        y_outer_test,
    ) = get_outer_data(
        dataset_name,
        fold_id,
    )

    # --------------------------------------------------
    # Remove sensitive attribute from model features.
    # --------------------------------------------------

    if sensitive_column is not None:
        if sensitive_column not in X_outer_train.columns:
            raise ValueError(
                f"Sensitive column '{sensitive_column}' "
                f"not found in dataset '{dataset_name}'."
            )

        sensitive_test = X_outer_test[
            sensitive_column
        ].copy()

        X_outer_train_model = (
            X_outer_train.drop(
                columns=[sensitive_column]
            ).copy()
        )

        X_outer_test_model = (
            X_outer_test.drop(
                columns=[sensitive_column]
            ).copy()
        )

    else:
        sensitive_test = None

        X_outer_train_model = (
            X_outer_train.copy()
        )

        X_outer_test_model = (
            X_outer_test.copy()
        )

    # --------------------------------------------------
    # Leakage-safe preprocessing.
    #
    # Fit on ALL outer training data.
    # Never fit on outer test data.
    # --------------------------------------------------

    (
        preprocessor,
        X_train_processed,
    ) = fit_transform_train(
        X_outer_train_model
    )

    X_test_processed = transform_data(
        preprocessor,
        X_outer_test_model,
    )

    # --------------------------------------------------
    # Build final model.
    # --------------------------------------------------

    model = build_model(
        model_name=model_name,
        params=params,
        random_state=random_state,
    )

    # --------------------------------------------------
    # Prepare target.
    # --------------------------------------------------

    if _is_xgboost(model_name):
        (
            target_encoder,
            y_outer_train_model,
            _,
            _,
        ) = _encode_binary_target(
            y_train=y_outer_train,
        )
    else:
        target_encoder = None
        y_outer_train_model = y_outer_train

    # --------------------------------------------------
    # Fit final model.
    # --------------------------------------------------

    model.fit(
        X_train_processed,
        y_outer_train_model,
    )

    # --------------------------------------------------
    # Predict untouched outer test set.
    # --------------------------------------------------

    y_pred_model = model.predict(
        X_test_processed
    )

    y_pred = _restore_predictions(
        model_name=model_name,
        predictions=y_pred_model,
        target_encoder=target_encoder,
    )

    # --------------------------------------------------
    # Prediction probabilities.
    # --------------------------------------------------

    y_prob = _get_prediction_probabilities(
        model=model,
        X=X_test_processed,
    )

    # --------------------------------------------------
    # Evaluate ONLY on the untouched outer test set.
    # --------------------------------------------------

    if sensitive_test is not None:
        metrics = evaluate_predictions(
            y_true=y_outer_test,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive=sensitive_test,
        )
    else:
        metrics = evaluate_predictions(
            y_true=y_outer_test,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive=["all"]
            * len(y_outer_test),
        )

    # --------------------------------------------------
    # Return benchmark result.
    # --------------------------------------------------

    return {
        "dataset": dataset_name,
        "outer_fold": fold_id,
        "model": model_name,
        **params,
        **metrics,
    }