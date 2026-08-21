from fair_hpo.optimizers.random_search import (
    generate_random_configs,
)

from fair_hpo.optimizers.bayesian_search import (
    create_bayesian_optimizer,
    ask_configuration,
    tell_result,
)

from fair_hpo.optimizers.baselines import (
    default_random_forest_params,
    default_xgboost_params,
)


def test_random_search():
    search_space = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 10],
    }

    configs = generate_random_configs(
        search_space=search_space,
        n_iter=5,
        random_state=42,
    )

    assert len(configs) == 5

    for config in configs:
        assert "n_estimators" in config
        assert "max_depth" in config


def test_bayesian_search():
    search_space = {
        "n_estimators": {
            "type": "integer",
            "low": 50,
            "high": 150,
        },
        "max_depth": {
            "type": "integer",
            "low": 3,
            "high": 10,
        },
        "learning_rate": {
            "type": "real",
            "low": 0.01,
            "high": 0.2,
            "prior": "log-uniform",
        },
    }

    optimizer = create_bayesian_optimizer(
        search_space=search_space,
        random_state=42,
    )

    config = ask_configuration(optimizer)

    assert "n_estimators" in config
    assert "max_depth" in config
    assert "learning_rate" in config

    tell_result(
        optimizer=optimizer,
        configuration=config,
        score=0.80,
    )


def test_baselines():
    rf = default_random_forest_params()
    xgb = default_xgboost_params()

    assert "n_estimators" in rf
    assert "max_depth" in rf

    assert "n_estimators" in xgb
    assert "learning_rate" in xgb


if __name__ == "__main__":
    test_random_search()
    test_bayesian_search()
    test_baselines()

    print("OPTIMIZER TESTS: PASS")