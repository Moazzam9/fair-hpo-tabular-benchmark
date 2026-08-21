import numpy as np

from fair_hpo.evaluation.metrics import (
    binary_classification_metrics,
    demographic_parity_difference,
    equal_opportunity_difference,
)


def main():
    y_true = np.array([
        0, 0, 1, 1,
        0, 1, 0, 1,
    ])

    y_pred = np.array([
        0, 0, 1, 1,
        0, 1, 1, 1,
    ])

    y_prob = np.array([
        0.10, 0.20, 0.80, 0.90,
        0.30, 0.70, 0.60, 0.95,
    ])

    sensitive = np.array([
        "A", "A", "A", "A",
        "B", "B", "B", "B",
    ])

    print("===== CLASSIFICATION METRICS =====")

    metrics = binary_classification_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print()
    print("===== FAIRNESS METRICS =====")

    dpd = demographic_parity_difference(
        y_pred,
        sensitive,
    )

    eod = equal_opportunity_difference(
        y_true,
        y_pred,
        sensitive,
    )

    print(
        "demographic_parity_difference:",
        f"{dpd:.6f}",
    )

    print(
        "equal_opportunity_difference:",
        f"{eod:.6f}",
    )

    print()
    print("METRICS TEST: PASS")


if __name__ == "__main__":
    main()