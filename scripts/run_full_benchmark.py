from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fair_hpo.config.loader import (
    load_dataset_config,
    load_fairness_config,
)
from fair_hpo.experiments.hpo import (
    run_bayesian_search,
    run_random_search,
)


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# BENCHMARK SETTINGS
# ------------------------------------------------------------

DATASETS = [
    "adult",
]

MODELS = [
    "random_forest",
    "xgboost",
]

OPTIMIZERS = [
    "random",
    "bayesian",
]

OUTER_FOLDS = [
    1,
]

N_ITER = 2

RANDOM_STATE = 42


# ------------------------------------------------------------
# SEARCH SPACES
# ------------------------------------------------------------

RANDOM_FOREST_RANDOM_SPACE = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


RANDOM_FOREST_BAYESIAN_SPACE = {
    "n_estimators": {
        "type": "integer",
        "low": 50,
        "high": 100,
    },
    "max_depth": {
        "type": "integer",
        "low": 3,
        "high": 20,
    },
    "min_samples_split": {
        "type": "integer",
        "low": 2,
        "high": 10,
    },
    "min_samples_leaf": {
        "type": "integer",
        "low": 1,
        "high": 4,
    },
}


XGBOOST_RANDOM_SPACE = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


XGBOOST_BAYESIAN_SPACE = {
    "n_estimators": {
        "type": "integer",
        "low": 50,
        "high": 100,
    },
    "max_depth": {
        "type": "integer",
        "low": 3,
        "high": 7,
    },
    "learning_rate": {
        "type": "real",
        "low": 0.03,
        "high": 0.1,
    },
    "subsample": {
        "type": "real",
        "low": 0.8,
        "high": 1.0,
    },
    "colsample_bytree": {
        "type": "real",
        "low": 0.8,
        "high": 1.0,
    },
    "min_child_weight": {
        "type": "integer",
        "low": 1,
        "high": 5,
    },
}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def get_search_space(
    model_name: str,
    optimizer: str,
) -> dict[str, Any]:
    if model_name == "random_forest":
        if optimizer == "random":
            return RANDOM_FOREST_RANDOM_SPACE

        return RANDOM_FOREST_BAYESIAN_SPACE

    if model_name == "xgboost":
        if optimizer == "random":
            return XGBOOST_RANDOM_SPACE

        return XGBOOST_BAYESIAN_SPACE

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def get_sensitive_column(
    dataset_name: str,
    fairness_config: dict[str, Any],
) -> str | None:

    dataset_config = fairness_config.get(
        "datasets",
        {},
    ).get(
        dataset_name,
        {},
    )

    enabled = dataset_config.get(
        "enabled",
        False,
    )

    if not enabled:
        return None

    return dataset_config.get(
        "sensitive_attribute"
    )


def run_one_experiment(
    dataset_name: str,
    fold_id: int,
    model_name: str,
    optimizer: str,
    sensitive_column: str | None,
) -> dict[str, Any]:

    search_space = get_search_space(
        model_name,
        optimizer,
    )

    print()
    print("=" * 80)
    print(
        f"DATASET={dataset_name} | "
        f"FOLD={fold_id} | "
        f"MODEL={model_name} | "
        f"OPTIMIZER={optimizer}"
    )
    print("=" * 80)

    start_time = time.perf_counter()

    if optimizer == "random":
        result = run_random_search(
            dataset_name=dataset_name,
            fold_id=fold_id,
            model_name=model_name,
            search_space=search_space,
            n_iter=N_ITER,
            sensitive_column=sensitive_column,
            objective="mean_balanced_accuracy",
            random_state=RANDOM_STATE,
        )

    elif optimizer == "bayesian":
        result = run_bayesian_search(
            dataset_name=dataset_name,
            fold_id=fold_id,
            model_name=model_name,
            search_space=search_space,
            n_iter=N_ITER,
            sensitive_column=sensitive_column,
            objective="mean_balanced_accuracy",
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer}"
        )

    runtime = time.perf_counter() - start_time

    outer_result = dict(
        result["outer_result"]
    )

    row = {
        "dataset": dataset_name,
        "outer_fold": fold_id,
        "model": model_name,
        "optimizer": optimizer,
        "runtime_seconds": runtime,
        "best_objective_score": result[
            "best_objective_score"
        ],
        "best_params": json.dumps(
            result["best_params"],
            default=str,
        ),
        **outer_result,
    }

    print()
    print("Best parameters:")
    print(result["best_params"])

    print()
    print("Outer metrics:")

    for key, value in outer_result.items():
        if key not in {
            "dataset",
            "outer_fold",
            "model",
        }:
            print(
                f"  {key}: {value}"
            )

    print()
    print(
        f"Runtime: {runtime:.2f} seconds"
    )

    return row


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 80)
    print("FULL HPO FAIRNESS BENCHMARK")
    print("=" * 80)

    dataset_config = load_dataset_config()
    fairness_config = load_fairness_config()

    print()
    print(
        "Configured datasets:",
        list(
            dataset_config.get(
                "datasets",
                {}
            ).keys()
        ),
    )

    all_results: list[dict[str, Any]] = []

    for dataset_name in DATASETS:

        if dataset_name not in dataset_config.get(
            "datasets",
            {}
        ):
            raise ValueError(
                f"Dataset '{dataset_name}' "
                f"is not defined in configs/datasets.yaml"
            )

        sensitive_column = get_sensitive_column(
            dataset_name,
            fairness_config,
        )

        print()
        print(
            f"Sensitive attribute for "
            f"{dataset_name}: "
            f"{sensitive_column}"
        )

        for fold_id in OUTER_FOLDS:

            for model_name in MODELS:

                for optimizer in OPTIMIZERS:

                    try:

                        row = run_one_experiment(
                            dataset_name=dataset_name,
                            fold_id=fold_id,
                            model_name=model_name,
                            optimizer=optimizer,
                            sensitive_column=sensitive_column,
                        )

                        all_results.append(row)

                    except Exception as exc:

                        print()
                        print(
                            "EXPERIMENT FAILED"
                        )
                        print(
                            f"Dataset: {dataset_name}"
                        )
                        print(
                            f"Fold: {fold_id}"
                        )
                        print(
                            f"Model: {model_name}"
                        )
                        print(
                            f"Optimizer: {optimizer}"
                        )
                        print(
                            f"Error: {exc}"
                        )

    if not all_results:
        raise RuntimeError(
            "No benchmark results were produced."
        )

    results_df = pd.DataFrame(
        all_results
    )

    csv_path = (
        RESULTS_DIR /
        "benchmark_results.csv"
    )

    json_path = (
        RESULTS_DIR /
        "benchmark_results.json"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
            default=str,
        )

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    print()
    print(
        results_df[
            [
                "dataset",
                "outer_fold",
                "model",
                "optimizer",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "roc_auc",
                "demographic_parity_difference",
                "equal_opportunity_difference",
                "runtime_seconds",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"CSV saved to: {csv_path}"
    )

    print(
        f"JSON saved to: {json_path}"
    )


if __name__ == "__main__":
    main()