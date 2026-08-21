from __future__ import annotations

import json
from pathlib import Path

from fair_hpo.experiments.hpo import (
    run_bayesian_search,
    run_random_search,
)


OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


RANDOM_SEARCH_SPACE = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


BAYESIAN_SEARCH_SPACE = {
    "n_estimators": {
        "type": "integer",
        "low": 50,
        "high": 200,
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


def save_result(name: str, result: dict):
    path = OUTPUT_DIR / f"{name}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Saved: {path}")


def main():
    dataset = "adult"
    fold_id = 1
    model = "random_forest"
    sensitive_column = "sex"

    n_iter = 10

    print("=" * 70)
    print("RANDOM SEARCH")
    print("=" * 70)

    random_result = run_random_search(
        dataset_name=dataset,
        fold_id=fold_id,
        model_name=model,
        search_space=RANDOM_SEARCH_SPACE,
        n_iter=n_iter,
        sensitive_column=sensitive_column,
        objective="mean_balanced_accuracy",
        random_state=42,
    )

    save_result(
        "adult_random_search",
        random_result,
    )

    print()
    print("Best random configuration:")
    print(random_result["best_params"])

    print()
    print("Best random inner metrics:")
    print(random_result["best_inner_metrics"])

    print()
    print("Outer result:")
    print(random_result["outer_result"])

    print()
    print("=" * 70)
    print("BAYESIAN SEARCH")
    print("=" * 70)

    bayesian_result = run_bayesian_search(
        dataset_name=dataset,
        fold_id=fold_id,
        model_name=model,
        search_space=BAYESIAN_SEARCH_SPACE,
        n_iter=n_iter,
        sensitive_column=sensitive_column,
        objective="mean_balanced_accuracy",
        random_state=42,
    )

    save_result(
        "adult_bayesian_search",
        bayesian_result,
    )

    print()
    print("Best Bayesian configuration:")
    print(bayesian_result["best_params"])

    print()
    print("Best Bayesian inner metrics:")
    print(bayesian_result["best_inner_metrics"])

    print()
    print("Outer result:")
    print(bayesian_result["outer_result"])


if __name__ == "__main__":
    main()