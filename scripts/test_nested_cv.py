import numpy as np

from fair_hpo.data.preprocessing import (
    fit_transform_train,
    transform_data,
)

from fair_hpo.evaluation.cv import (
    get_inner_cv,
    get_outer_data,
)

from fair_hpo.models.factory import (
    build_model,
)


def test_one_outer_fold():
    dataset = "wdbc"
    fold_id = 1

    print(
        f"\n===== NESTED CV TEST: "
        f"{dataset.upper()} FOLD {fold_id} ====="
    )

    (
        X_outer_train,
        X_outer_test,
        y_outer_train,
        y_outer_test,
    ) = get_outer_data(
        dataset,
        fold_id,
    )

    print(
        "Outer train:",
        X_outer_train.shape,
    )

    print(
        "Outer test:",
        X_outer_test.shape,
    )

    # --------------------------------------------------
    # FIT PREPROCESSING ONLY ON OUTER TRAIN
    # --------------------------------------------------

    preprocessor, X_train_processed = (
        fit_transform_train(
            X_outer_train
        )
    )

    X_test_processed = transform_data(
        preprocessor,
        X_outer_test,
    )

    print(
        "Processed outer train:",
        X_train_processed.shape,
    )

    print(
        "Processed outer test:",
        X_test_processed.shape,
    )

    # --------------------------------------------------
    # INNER CV
    # --------------------------------------------------

    inner_splits = get_inner_cv(
        X_outer_train,
        y_outer_train,
        random_state=42,
    )

    print(
        f"Number of inner folds: "
        f"{len(inner_splits)}"
    )

    # --------------------------------------------------
    # TEST ONE MODEL FIT
    # --------------------------------------------------

    train_idx, validation_idx = inner_splits[0]

    X_inner_train = (
        X_train_processed.iloc[train_idx]
    )

    X_inner_validation = (
        X_train_processed.iloc[validation_idx]
    )

    y_inner_train = (
        y_outer_train.iloc[train_idx]
    )

    y_inner_validation = (
        y_outer_train.iloc[validation_idx]
    )

    model = build_model(
        "random_forest",
        params={
            "n_estimators": 10,
            "max_depth": 5,
        },
        random_state=1000,
    )

    model.fit(
        X_inner_train,
        y_inner_train,
    )

    probabilities = model.predict_proba(
        X_inner_validation
    )[:, 1]

    print(
        "Validation predictions:",
        len(probabilities),
    )

    print(
        "Probability range:",
        float(np.min(probabilities)),
        "to",
        float(np.max(probabilities)),
    )

    # --------------------------------------------------
    # FINAL OUTER TEST IS STILL UNTOUCHED BY MODEL
    # --------------------------------------------------

    assert len(X_test_processed) == len(
        y_outer_test
    )

    assert (
        X_train_processed.shape[1]
        == X_test_processed.shape[1]
    )

    assert not X_train_processed.isna().any().any()
    assert not X_test_processed.isna().any().any()

    print(
        "\nNESTED CV STRUCTURE TEST: PASS"
    )


if __name__ == "__main__":
    test_one_outer_fold()