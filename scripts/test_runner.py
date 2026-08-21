from fair_hpo.experiments.runner import (
    evaluate_configuration_inner_cv,
    fit_final_outer_model,
)


def test_inner_cv_runner():
    result = evaluate_configuration_inner_cv(
        dataset_name="adult",
        fold_id=1,
        model_name="random_forest",
        params={
            "n_estimators": 10,
            "max_depth": 5,
        },
        sensitive_column="sex",
    )

    assert "mean_accuracy" in result
    assert "mean_balanced_accuracy" in result
    assert "mean_f1" in result
    assert "mean_roc_auc" in result
    assert "mean_demographic_parity_difference" in result
    assert "mean_equal_opportunity_difference" in result


def test_outer_evaluation():
    result = fit_final_outer_model(
        dataset_name="adult",
        fold_id=1,
        model_name="random_forest",
        params={
            "n_estimators": 10,
            "max_depth": 5,
        },
        sensitive_column="sex",
    )

    assert result["dataset"] == "adult"
    assert result["outer_fold"] == 1
    assert result["model"] == "random_forest"

    assert "accuracy" in result
    assert "balanced_accuracy" in result
    assert "f1" in result
    assert "roc_auc" in result
    assert "demographic_parity_difference" in result
    assert "equal_opportunity_difference" in result