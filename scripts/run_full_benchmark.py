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

# Dataset names are loaded automatically from configs/datasets.yaml.
# This prevents the benchmark from accidentally running only one
# dataset when additional datasets are configured.
OUTER_FOLDS = [
    1,
]

MODELS = [
    "random_forest",
    "xgboost",
]

OPTIMIZERS = [
    "random",
    "bayesian",
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
    """
    Return the search space for a model/optimizer combination.
    """

    if model_name == "random_forest":
        if optimizer == "random":
            return RANDOM_FOREST_RANDOM_SPACE

        if optimizer == "bayesian":
            return RANDOM_FOREST_BAYESIAN_SPACE

    elif model_name == "xgboost":
        if optimizer == "random":
            return XGBOOST_RANDOM_SPACE

        if optimizer == "bayesian":
            return XGBOOST_BAYESIAN_SPACE

    raise ValueError(
        f"Unknown model/optimizer combination: "
        f"{model_name}/{optimizer}"
    )


def get_sensitive_column(
    dataset_name: str,
    fairness_config: dict[str, Any],
) -> str | None:
    """
    Get the sensitive attribute configured for a dataset.

    If fairness evaluation is disabled for the dataset, return None.
    """

    dataset_config = (
        fairness_config
        .get("datasets", {})
        .get(dataset_name, {})
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


def get_configured_datasets(
    dataset_config: dict[str, Any],
) -> list[str]:
    """
    Return every dataset defined in configs/datasets.yaml.

    The order in the YAML file is preserved.
    """

    datasets = dataset_config.get(
        "datasets",
        {},
    )

    if not datasets:
        raise ValueError(
            "No datasets are configured in "
            "configs/datasets.yaml."
        )

    return list(datasets.keys())


def run_one_experiment(
    dataset_name: str,
    fold_id: int,
    model_name: str,
    optimizer: str,
    sensitive_column: str | None,
) -> dict[str, Any]:
    """
    Run one complete nested-HPO experiment.
    """

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


def save_results(
    all_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Save benchmark results to CSV and JSON.
    """

    if not all_results:
        raise RuntimeError(
            "No benchmark results were produced."
        )

    results_df = pd.DataFrame(
        all_results
    )

    csv_path = (
        RESULTS_DIR
        / "benchmark_results.csv"
    )

    json_path = (
        RESULTS_DIR
        / "benchmark_results.json"
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
    print(
        f"CSV saved to: {csv_path}"
    )

    print(
        f"JSON saved to: {json_path}"
    )

    return results_df


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    """
    Run the complete configured benchmark.

    Every configured dataset is evaluated with:

        random_forest + random
        random_forest + bayesian
        xgboost + random
        xgboost + bayesian

    Each experiment is isolated so a failure does not terminate
    the remaining benchmark.
    """

    print("=" * 80)
    print("FULL HPO FAIRNESS BENCHMARK")
    print("=" * 80)

    # --------------------------------------------------------
    # Load configuration.
    # --------------------------------------------------------

    dataset_config = load_dataset_config()
    fairness_config = load_fairness_config()

    datasets = get_configured_datasets(
        dataset_config
    )

    print()
    print(
        "Configured datasets:",
        datasets,
    )

    print()
    print(
        "Models:",
        MODELS,
    )

    print()
    print(
        "Optimizers:",
        OPTIMIZERS,
    )

    print()
    print(
        "Outer folds:",
        OUTER_FOLDS,
    )

    print()
    print(
        "HPO iterations:",
        N_ITER,
    )

    # --------------------------------------------------------
    # Validate dataset configuration before starting.
    # --------------------------------------------------------

    configured_dataset_map = dataset_config.get(
        "datasets",
        {},
    )

    for dataset_name in datasets:
        if dataset_name not in configured_dataset_map:
            raise ValueError(
                f"Dataset '{dataset_name}' "
                f"is not defined in configs/datasets.yaml"
            )

    # --------------------------------------------------------
    # Run benchmark.
    # --------------------------------------------------------

    all_results: list[dict[str, Any]] = []

    total_experiments = (
        len(datasets)
        * len(OUTER_FOLDS)
        * len(MODELS)
        * len(OPTIMIZERS)
    )

    experiment_number = 0

    for dataset_name in datasets:

        sensitive_column = get_sensitive_column(
            dataset_name,
            fairness_config,
        )

        print()
        print("-" * 80)
        print(
            f"Dataset: {dataset_name}"
        )
        print(
            f"Sensitive attribute: "
            f"{sensitive_column}"
        )
        print("-" * 80)

        for fold_id in OUTER_FOLDS:

            for model_name in MODELS:

                for optimizer in OPTIMIZERS:

                    experiment_number += 1

                    print()
                    print(
                        f"Experiment "
                        f"{experiment_number}/"
                        f"{total_experiments}"
                    )

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
                        print("=" * 80)
                        print("EXPERIMENT FAILED")
                        print("=" * 80)

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

                        print(
                            "Continuing with the next experiment..."
                        )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    results_df = save_results(
        all_results
    )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Successful experiments: "
        f"{len(all_results)}/{total_experiments}"
    )

    print()

    summary_columns = [
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

    available_columns = [
        column
        for column in summary_columns
        if column in results_df.columns
    ]

    print(
        results_df[
            available_columns
        ].to_string(index=False)
    )

    print()
    print(
        f"CSV saved to: "
        f"{RESULTS_DIR / 'benchmark_results.csv'}"
    )

    print(
        f"JSON saved to: "
        f"{RESULTS_DIR / 'benchmark_results.json'}"
    )


if __name__ == "__main__":
    main()