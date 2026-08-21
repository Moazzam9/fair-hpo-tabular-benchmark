from fair_hpo.experiments.hpo import (
    run_bayesian_search,
    run_random_search,
)


SEARCH_SPACE = {
    "n_estimators": {
        "type": "integer",
        "low": 5,
        "high": 10,
    },
    "max_depth": {
        "type": "integer",
        "low": 3,
        "high": 5,
    },
}


def test_random_hpo():
    result = run_random_search(
        dataset_name="adult",
        fold_id=1,
        model_name="random_forest",
        search_space={
            "n_estimators": [5, 10],
            "max_depth": [3, 5],
        },
        n_iter=2,
        sensitive_column="sex",
        objective="mean_balanced_accuracy",
        random_state=42,
    )

    assert result["best_params"] is not None
    assert result["best_inner_metrics"] is not None
    assert result["outer_result"]["dataset"] == "adult"

    assert len(result["history"]) == 2


def test_bayesian_hpo():
    result = run_bayesian_search(
        dataset_name="adult",
        fold_id=1,
        model_name="random_forest",
        search_space=SEARCH_SPACE,
        n_iter=2,
        sensitive_column="sex",
        objective="mean_balanced_accuracy",
        random_state=42,
    )

    assert result["best_params"] is not None
    assert result["best_inner_metrics"] is not None
    assert result["outer_result"]["dataset"] == "adult"

    assert len(result["history"]) == 2