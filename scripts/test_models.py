from fair_hpo.models.factory import build_model


def test_random_forest():
    print("\n===== TESTING RANDOM FOREST =====")

    params = {
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }

    model = build_model(
        model_name="random_forest",
        params=params,
        random_state=1000,
    )

    print(model)

    assert model.n_estimators == 50
    assert model.max_depth == 10
    assert model.min_samples_split == 2
    assert model.min_samples_leaf == 1
    assert model.max_features == "sqrt"
    assert model.n_jobs == 1
    assert model.random_state == 1000

    print("Random Forest: PASS")


def test_xgboost():
    print("\n===== TESTING XGBOOST =====")

    params = {
        "n_estimators": 50,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    }

    model = build_model(
        model_name="xgboost",
        params=params,
        random_state=1000,
    )

    print(model)

    assert model.n_estimators == 50
    assert model.max_depth == 6
    assert model.learning_rate == 0.1
    assert model.subsample == 0.8
    assert model.colsample_bytree == 0.8
    assert model.min_child_weight == 1
    assert model.n_jobs == 1
    assert model.random_state == 1000

    print("XGBoost: PASS")


if __name__ == "__main__":
    test_random_forest()
    test_xgboost()

    print("\nAll model factory tests passed.")