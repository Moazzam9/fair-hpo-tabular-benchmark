"""
Generate publication-ready statistical analysis for the fairness-aware HPO benchmark.

Methodology
-----------
The benchmark contains:

    3 datasets × 2 models × 2 optimizers × 5 outer folds = 60 observations.

All comparisons are paired at the outer-fold level.

Model comparison:
    Random Forest vs XGBoost
    Pairing keys = dataset, optimizer, outer_fold
    Expected pairs per metric = 3 × 2 × 5 = 30

Optimizer comparison:
    Random Search vs Bayesian Optimization
    Pairing keys = dataset, model, outer_fold
    Expected pairs per metric = 3 × 2 × 5 = 30

Adult fairness comparisons:
    Pairing keys follow the same logic but are restricted to Adult,
    because Adult is currently the only dataset with a configured
    sensitive attribute.

No model training, HPO, data downloading, or modification of the
original benchmark result files is performed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "benchmark_results.csv"
OUTPUT_DIR = ROOT / "results" / "publication"

STAT_TESTS = OUTPUT_DIR / "statistical_tests.csv"
EFFECT_SIZES = OUTPUT_DIR / "effect_sizes.csv"
SUMMARY_MD = OUTPUT_DIR / "statistical_summary.md"


# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------

DATASETS = ["wdbc", "adult", "bank_marketing"]
MODELS = ["random_forest", "xgboost"]
OPTIMIZERS = ["random", "bayesian"]
FOLDS = [1, 2, 3, 4, 5]

PERFORMANCE_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
]

FAIRNESS_METRICS = [
    "demographic_parity_difference",
    "equal_opportunity_difference",
]

RUNTIME_METRIC = "runtime_seconds"

ALL_METRICS = PERFORMANCE_METRICS + FAIRNESS_METRICS + [RUNTIME_METRIC]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def fail(message: str) -> None:
    """Print an error and terminate with a non-zero exit status."""
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def clean_number(value):
    """Convert numeric output to a regular Python float where possible."""
    if value is None:
        return np.nan

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def fmt(value, digits: int = 4) -> str:
    """Format a numeric value for the Markdown report."""
    value = clean_number(value)

    if pd.isna(value):
        return "NA"

    return f"{value:.{digits}f}"


def fmt_p(value) -> str:
    """Format a p-value."""
    value = clean_number(value)

    if pd.isna(value):
        return "NA"

    if value < 0.001:
        return "<0.001"

    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def load_and_validate() -> pd.DataFrame:
    """Load benchmark results and validate the complete 60-row design."""

    if not INPUT.exists():
        fail(f"Input file does not exist: {INPUT}")

    df = pd.read_csv(INPUT)

    if len(df) != 60:
        fail(
            f"Expected exactly 60 rows, found {len(df)}."
        )

    required_columns = {
        "dataset",
        "model",
        "optimizer",
        "outer_fold",
        *ALL_METRICS,
    }

    missing = sorted(required_columns - set(df.columns))

    if missing:
        fail(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # Normalize strings.
    for column in ["dataset", "model", "optimizer"]:
        df[column] = df[column].astype(str).str.strip().str.lower()

    df["outer_fold"] = pd.to_numeric(
        df["outer_fold"],
        errors="coerce",
    )

    if df["outer_fold"].isna().any():
        fail("outer_fold contains non-numeric values.")

    df["outer_fold"] = df["outer_fold"].astype(int)

    # Validate dimensions.
    if sorted(df["dataset"].unique()) != sorted(DATASETS):
        fail(
            "Dataset dimension mismatch. "
            f"Expected {DATASETS}, found {sorted(df['dataset'].unique())}."
        )

    if sorted(df["model"].unique()) != sorted(MODELS):
        fail(
            "Model dimension mismatch. "
            f"Expected {MODELS}, found {sorted(df['model'].unique())}."
        )

    if sorted(df["optimizer"].unique()) != sorted(OPTIMIZERS):
        fail(
            "Optimizer dimension mismatch. "
            f"Expected {OPTIMIZERS}, found {sorted(df['optimizer'].unique())}."
        )

    if sorted(df["outer_fold"].unique()) != FOLDS:
        fail(
            "Outer-fold dimension mismatch. "
            f"Expected {FOLDS}, found {sorted(df['outer_fold'].unique())}."
        )

    # Validate all numeric metrics.
    for metric in ALL_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

        if df[metric].isna().all():
            fail(f"Metric '{metric}' contains no usable numeric values.")

    # The full factorial design must contain exactly one row per combination.
    dimensions = [
        "dataset",
        "model",
        "optimizer",
        "outer_fold",
    ]

    duplicated = df.duplicated(dimensions, keep=False)

    if duplicated.any():
        bad = df.loc[duplicated, dimensions]
        fail(
            "Duplicate experiment combinations detected:\n"
            + bad.to_string(index=False)
        )

    expected_groups = 3 * 2 * 2

    actual_groups = (
        df[["dataset", "model", "optimizer"]]
        .drop_duplicates()
        .shape[0]
    )

    if actual_groups != expected_groups:
        fail(
            f"Expected {expected_groups} experiment groups, "
            f"found {actual_groups}."
        )

    expected_rows_per_group = 5

    group_counts = (
        df.groupby(
            ["dataset", "model", "optimizer"],
            dropna=False,
        )
        .size()
    )

    if not (group_counts == expected_rows_per_group).all():
        fail(
            "Each dataset/model/optimizer group must contain "
            "exactly five outer folds."
        )

    print(
        "[OK] Validation passed: "
        f"{len(df)} rows, "
        f"{actual_groups} experiment groups, "
        "5 outer folds per group."
    )

    return df


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def get_metric_pairs(
    df: pd.DataFrame,
    metric: str,
    left_filter: dict,
    right_filter: dict,
    pair_keys: list[str],
    expected_pairs: int,
    comparison_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Construct paired observations.

    Crucially, pairing is done using the complete experimental identity.

    Model comparison:
        dataset + optimizer + outer_fold

    Optimizer comparison:
        dataset + model + outer_fold
    """

    left = df.copy()
    right = df.copy()

    for column, value in left_filter.items():
        left = left[left[column] == value]

    for column, value in right_filter.items():
        right = right[right[column] == value]

    left = left[pair_keys + [metric]].copy()
    right = right[pair_keys + [metric]].copy()

    left = left.rename(columns={metric: "left_value"})
    right = right.rename(columns={metric: "right_value"})

    # Defensive uniqueness validation.
    if left.duplicated(pair_keys).any():
        fail(
            f"Left side of {comparison_name} contains duplicate "
            f"pair keys for metric '{metric}'."
        )

    if right.duplicated(pair_keys).any():
        fail(
            f"Right side of {comparison_name} contains duplicate "
            f"pair keys for metric '{metric}'."
        )

    paired = pd.merge(
        left,
        right,
        on=pair_keys,
        how="inner",
        validate="one_to_one",
    )

    # Only retain pairs where both observations are available.
    paired = paired.dropna(
        subset=["left_value", "right_value"]
    ).reset_index(drop=True)

    if len(paired) != expected_pairs:
        fail(
            f"Expected {expected_pairs} paired observations for "
            f"{comparison_name} of '{metric}', found {len(paired)}."
        )

    differences = (
        paired["right_value"].to_numpy(dtype=float)
        - paired["left_value"].to_numpy(dtype=float)
    )

    return paired, differences


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_statistics(
    left: np.ndarray,
    right: np.ndarray,
) -> dict:
    """
    Calculate paired statistical tests and effect sizes.

    Tests:
        - paired t-test
        - Wilcoxon signed-rank test

    Effect:
        - Cohen's dz based on paired differences

    The Wilcoxon test is reported as the primary non-parametric test
    because each experimental comparison is paired by outer fold.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    mask = np.isfinite(left) & np.isfinite(right)

    left = left[mask]
    right = right[mask]

    n = len(left)

    if n == 0:
        return {
            "n": 0,
            "mean_left": np.nan,
            "mean_right": np.nan,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "t_statistic": np.nan,
            "t_pvalue": np.nan,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_pvalue": np.nan,
            "cohens_dz": np.nan,
        }

    differences = right - left

    mean_difference = float(np.mean(differences))
    median_difference = float(np.median(differences))

    # Paired t-test.
    if n >= 2 and np.std(differences, ddof=1) > 0:
        t_result = stats.ttest_rel(left, right)

        t_statistic = clean_number(t_result.statistic)
        t_pvalue = clean_number(t_result.pvalue)

        sd_difference = float(
            np.std(differences, ddof=1)
        )

        cohens_dz = (
            mean_difference / sd_difference
            if sd_difference > 0
            else np.nan
        )
    else:
        t_statistic = np.nan
        t_pvalue = np.nan
        cohens_dz = np.nan

    # Wilcoxon.
    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0
    elif len(nonzero) >= 2:
        try:
            result = stats.wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            wilcoxon_statistic = clean_number(
                result.statistic
            )
            wilcoxon_pvalue = clean_number(
                result.pvalue
            )

        except ValueError:
            wilcoxon_statistic = np.nan
            wilcoxon_pvalue = np.nan
    else:
        wilcoxon_statistic = np.nan
        wilcoxon_pvalue = np.nan

    return {
        "n": n,
        "mean_left": float(np.mean(left)),
        "mean_right": float(np.mean(right)),
        "mean_difference": mean_difference,
        "median_difference": median_difference,
        "t_statistic": t_statistic,
        "t_pvalue": t_pvalue,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "cohens_dz": cohens_dz,
    }


def effect_interpretation(value) -> str:
    """Interpret absolute Cohen's dz using conventional thresholds."""

    value = clean_number(value)

    if pd.isna(value):
        return "not estimable"

    magnitude = abs(value)

    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"

    return "large"


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def run_model_comparisons(
    df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """
    Compare Random Forest against XGBoost.

    Pairing keys:

        dataset + optimizer + outer_fold

    Therefore each metric has:

        3 datasets × 2 optimizers × 5 folds = 30 pairs.
    """

    tests = []
    effects = []

    pair_keys = [
        "dataset",
        "optimizer",
        "outer_fold",
    ]

    expected_pairs = (
        len(DATASETS)
        * len(OPTIMIZERS)
        * len(FOLDS)
    )

    for metric in PERFORMANCE_METRICS + [RUNTIME_METRIC]:

        paired, _ = get_metric_pairs(
            df=df,
            metric=metric,
            left_filter={"model": "random_forest"},
            right_filter={"model": "xgboost"},
            pair_keys=pair_keys,
            expected_pairs=expected_pairs,
            comparison_name="model comparison",
        )

        result = paired_statistics(
            paired["left_value"].to_numpy(),
            paired["right_value"].to_numpy(),
        )

        base = {
            "comparison": "model",
            "scope": "all_datasets",
            "metric": metric,
            "left_method": "random_forest",
            "right_method": "xgboost",
            "pairing": "dataset + optimizer + outer_fold",
        }

        tests.append(
            {
                **base,
                **result,
            }
        )

        effects.append(
            {
                **base,
                "effect_size": result["cohens_dz"],
                "effect_size_name": "Cohen's dz",
                "effect_interpretation": effect_interpretation(
                    result["cohens_dz"]
                ),
            }
        )

    return tests, effects


# ---------------------------------------------------------------------------
# Optimizer comparison
# ---------------------------------------------------------------------------

def run_optimizer_comparisons(
    df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """
    Compare Random Search against Bayesian Optimization.

    Pairing keys:

        dataset + model + outer_fold

    Therefore each metric has:

        3 datasets × 2 models × 5 folds = 30 pairs.
    """

    tests = []
    effects = []

    pair_keys = [
        "dataset",
        "model",
        "outer_fold",
    ]

    expected_pairs = (
        len(DATASETS)
        * len(MODELS)
        * len(FOLDS)
    )

    for metric in PERFORMANCE_METRICS + [RUNTIME_METRIC]:

        paired, _ = get_metric_pairs(
            df=df,
            metric=metric,
            left_filter={"optimizer": "random"},
            right_filter={"optimizer": "bayesian"},
            pair_keys=pair_keys,
            expected_pairs=expected_pairs,
            comparison_name="optimizer comparison",
        )

        result = paired_statistics(
            paired["left_value"].to_numpy(),
            paired["right_value"].to_numpy(),
        )

        base = {
            "comparison": "optimizer",
            "scope": "all_datasets",
            "metric": metric,
            "left_method": "random",
            "right_method": "bayesian",
            "pairing": "dataset + model + outer_fold",
        }

        tests.append(
            {
                **base,
                **result,
            }
        )

        effects.append(
            {
                **base,
                "effect_size": result["cohens_dz"],
                "effect_size_name": "Cohen's dz",
                "effect_interpretation": effect_interpretation(
                    result["cohens_dz"]
                ),
            }
        )

    return tests, effects


# ---------------------------------------------------------------------------
# Dataset-specific comparisons
# ---------------------------------------------------------------------------

def run_dataset_specific_comparisons(
    df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """
    Produce dataset-specific paired comparisons.

    This is useful for publication interpretation because an overall
    paired comparison can hide dataset-specific behavior.

    Model comparison:
        2 optimizers × 5 folds = 10 pairs per dataset.

    Optimizer comparison:
        2 models × 5 folds = 10 pairs per dataset.
    """

    tests = []
    effects = []

    # Model comparison within each dataset.
    for dataset in DATASETS:

        for metric in PERFORMANCE_METRICS + [RUNTIME_METRIC]:

            pair_keys = [
                "optimizer",
                "outer_fold",
            ]

            paired, _ = get_metric_pairs(
                df=df,
                metric=metric,
                left_filter={
                    "dataset": dataset,
                    "model": "random_forest",
                },
                right_filter={
                    "dataset": dataset,
                    "model": "xgboost",
                },
                pair_keys=pair_keys,
                expected_pairs=10,
                comparison_name=(
                    f"model comparison for {dataset}"
                ),
            )

            result = paired_statistics(
                paired["left_value"].to_numpy(),
                paired["right_value"].to_numpy(),
            )

            base = {
                "comparison": "model",
                "scope": dataset,
                "metric": metric,
                "left_method": "random_forest",
                "right_method": "xgboost",
                "pairing": "optimizer + outer_fold",
            }

            tests.append(
                {
                    **base,
                    **result,
                }
            )

            effects.append(
                {
                    **base,
                    "effect_size": result["cohens_dz"],
                    "effect_size_name": "Cohen's dz",
                    "effect_interpretation": effect_interpretation(
                        result["cohens_dz"]
                    ),
                }
            )

    # Optimizer comparison within each dataset.
    for dataset in DATASETS:

        for metric in PERFORMANCE_METRICS + [RUNTIME_METRIC]:

            pair_keys = [
                "model",
                "outer_fold",
            ]

            paired, _ = get_metric_pairs(
                df=df,
                metric=metric,
                left_filter={
                    "dataset": dataset,
                    "optimizer": "random",
                },
                right_filter={
                    "dataset": dataset,
                    "optimizer": "bayesian",
                },
                pair_keys=pair_keys,
                expected_pairs=10,
                comparison_name=(
                    f"optimizer comparison for {dataset}"
                ),
            )

            result = paired_statistics(
                paired["left_value"].to_numpy(),
                paired["right_value"].to_numpy(),
            )

            base = {
                "comparison": "optimizer",
                "scope": dataset,
                "metric": metric,
                "left_method": "random",
                "right_method": "bayesian",
                "pairing": "model + outer_fold",
            }

            tests.append(
                {
                    **base,
                    **result,
                }
            )

            effects.append(
                {
                    **base,
                    "effect_size": result["cohens_dz"],
                    "effect_size_name": "Cohen's dz",
                    "effect_interpretation": effect_interpretation(
                        result["cohens_dz"]
                    ),
                }
            )

    return tests, effects


# ---------------------------------------------------------------------------
# Fairness analysis
# ---------------------------------------------------------------------------

def run_fairness_comparisons(
    df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """
    Analyze Adult fairness metrics.

    Adult is the only currently configured fairness dataset.

    Model comparison:
        Random Forest vs XGBoost
        10 paired observations per fairness metric.

    Optimizer comparison:
        Random vs Bayesian
        10 paired observations per fairness metric.

    WDBC and Bank Marketing are deliberately excluded because their
    fairness metrics are not applicable without configured sensitive
    attributes.
    """

    adult = df[df["dataset"] == "adult"].copy()

    tests = []
    effects = []

    for metric in FAIRNESS_METRICS:

        # Model.
        paired, _ = get_metric_pairs(
            df=adult,
            metric=metric,
            left_filter={"model": "random_forest"},
            right_filter={"model": "xgboost"},
            pair_keys=["optimizer", "outer_fold"],
            expected_pairs=10,
            comparison_name=(
                f"Adult fairness model comparison"
            ),
        )

        result = paired_statistics(
            paired["left_value"].to_numpy(),
            paired["right_value"].to_numpy(),
        )

        base = {
            "comparison": "fairness_model",
            "scope": "adult",
            "metric": metric,
            "left_method": "random_forest",
            "right_method": "xgboost",
            "pairing": "optimizer + outer_fold",
        }

        tests.append(
            {
                **base,
                **result,
            }
        )

        effects.append(
            {
                **base,
                "effect_size": result["cohens_dz"],
                "effect_size_name": "Cohen's dz",
                "effect_interpretation": effect_interpretation(
                    result["cohens_dz"]
                ),
            }
        )

        # Optimizer.
        paired, _ = get_metric_pairs(
            df=adult,
            metric=metric,
            left_filter={"optimizer": "random"},
            right_filter={"optimizer": "bayesian"},
            pair_keys=["model", "outer_fold"],
            expected_pairs=10,
            comparison_name=(
                f"Adult fairness optimizer comparison"
            ),
        )

        result = paired_statistics(
            paired["left_value"].to_numpy(),
            paired["right_value"].to_numpy(),
        )

        base = {
            "comparison": "fairness_optimizer",
            "scope": "adult",
            "metric": metric,
            "left_method": "random",
            "right_method": "bayesian",
            "pairing": "model + outer_fold",
        }

        tests.append(
            {
                **base,
                **result,
            }
        )

        effects.append(
            {
                **base,
                "effect_size": result["cohens_dz"],
                "effect_size_name": "Cohen's dz",
                "effect_interpretation": effect_interpretation(
                    result["cohens_dz"]
                ),
            }
        )

    return tests, effects


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def create_markdown_summary(
    df: pd.DataFrame,
    tests: pd.DataFrame,
    effects: pd.DataFrame,
) -> None:
    """Create a concise publication-oriented statistical summary."""

    lines = []

    lines.append("# Statistical Analysis — Fairness-Aware HPO Benchmark")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        f"The benchmark contains **{len(df)} fold-level observations** "
        f"covering **3 datasets × 2 models × 2 optimizers × 5 outer folds**."
    )
    lines.append("")
    lines.append(
        "All primary comparisons are paired at the outer-fold level. "
        "The statistical analysis uses paired t-tests, Wilcoxon "
        "signed-rank tests, and Cohen's dz effect sizes."
    )
    lines.append("")

    lines.append("## Pairing Methodology")
    lines.append("")
    lines.append(
        "- **Model comparison:** Random Forest vs XGBoost, paired by "
        "dataset + optimizer + outer fold; 30 pairs per metric."
    )
    lines.append(
        "- **Optimizer comparison:** Random Search vs Bayesian "
        "Optimization, paired by dataset + model + outer fold; "
        "30 pairs per metric."
    )
    lines.append(
        "- **Adult fairness comparisons:** paired using the corresponding "
        "experimental dimensions; 10 pairs per fairness metric."
    )
    lines.append("")

    lines.append("## Primary Statistical Results")
    lines.append("")

    primary = tests[
        tests["scope"] == "all_datasets"
    ].copy()

    for comparison in ["model", "optimizer"]:

        subset = primary[
            primary["comparison"] == comparison
        ]

        if subset.empty:
            continue

        if comparison == "model":
            title = "### Random Forest vs XGBoost"
        else:
            title = "### Random Search vs Bayesian Optimization"

        lines.append(title)
        lines.append("")

        for _, row in subset.iterrows():

            lines.append(
                f"- **{row['metric']}**: "
                f"{row['left_method']} mean = "
                f"{fmt(row['mean_left'])}, "
                f"{row['right_method']} mean = "
                f"{fmt(row['mean_right'])}, "
                f"mean difference = "
                f"{fmt(row['mean_difference'])}, "
                f"Wilcoxon p = "
                f"{fmt_p(row['wilcoxon_pvalue'])}, "
                f"Cohen's dz = "
                f"{fmt(row['cohens_dz'])}."
            )

        lines.append("")

    lines.append("## Dataset-Specific Interpretation")
    lines.append("")

    dataset_tests = tests[
        tests["scope"].isin(DATASETS)
    ].copy()

    for dataset in DATASETS:

        lines.append(f"### {dataset}")
        lines.append("")

        subset = dataset_tests[
            dataset_tests["scope"] == dataset
        ]

        for comparison in [
            "model",
            "optimizer",
        ]:

            comp = subset[
                subset["comparison"] == comparison
            ]

            if comp.empty:
                continue

            if comparison == "model":
                label = "Model comparison"
            else:
                label = "Optimizer comparison"

            lines.append(f"**{label}:**")

            for _, row in comp.iterrows():

                lines.append(
                    f"- {row['metric']}: "
                    f"mean difference = "
                    f"{fmt(row['mean_difference'])}, "
                    f"Wilcoxon p = "
                    f"{fmt_p(row['wilcoxon_pvalue'])}, "
                    f"Cohen's dz = "
                    f"{fmt(row['cohens_dz'])} "
                    f"({row.get('effect_interpretation', 'see effect_sizes.csv')})."
                )

            lines.append("")

    lines.append("## Fairness Analysis")
    lines.append("")
    lines.append(
        "Fairness inference is restricted to **Adult**, where the "
        "configured sensitive attribute is `sex`."
    )
    lines.append("")
    lines.append(
        "WDBC and Bank Marketing are excluded from inferential fairness "
        "comparisons because no sensitive attribute is currently configured."
    )
    lines.append("")

    fairness_tests = tests[
        tests["comparison"].isin(
            ["fairness_model", "fairness_optimizer"]
        )
    ]

    for _, row in fairness_tests.iterrows():

        lines.append(
            f"- **{row['comparison']} / {row['metric']}**: "
            f"mean difference = {fmt(row['mean_difference'])}, "
            f"Wilcoxon p = {fmt_p(row['wilcoxon_pvalue'])}, "
            f"Cohen's dz = {fmt(row['cohens_dz'])}."
        )

    lines.append("")
    lines.append("## Interpretation Guidelines")
    lines.append("")
    lines.append(
        "A p-value below 0.05 is treated as evidence against the "
        "null hypothesis for the corresponding paired test, but "
        "statistical significance should not be interpreted as "
        "practical importance without considering the effect size "
        "and magnitude of the observed difference."
    )
    lines.append("")
    lines.append(
        "Cohen's dz is interpreted using conventional descriptive "
        "thresholds: <0.2 negligible, 0.2–<0.5 small, "
        "0.5–<0.8 medium, and ≥0.8 large."
    )
    lines.append("")
    lines.append(
        "No multiple-comparison correction is applied automatically. "
        "The resulting p-values should therefore be treated as "
        "exploratory unless a prespecified multiplicity procedure "
        "is adopted for the final publication."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "This analysis reads the existing "
        "`results/benchmark_results.csv` file only. It does not "
        "perform model training, HPO, or dataset downloading and "
        "does not modify the benchmark result files."
    )
    lines.append("")

    SUMMARY_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("[INFO] Running paired statistical comparisons...")

    all_tests = []
    all_effects = []

    # Overall comparisons.
    tests, effects = run_model_comparisons(df)
    all_tests.extend(tests)
    all_effects.extend(effects)

    tests, effects = run_optimizer_comparisons(df)
    all_tests.extend(tests)
    all_effects.extend(effects)

    # Dataset-specific comparisons.
    tests, effects = run_dataset_specific_comparisons(df)
    all_tests.extend(tests)
    all_effects.extend(effects)

    # Fairness.
    tests, effects = run_fairness_comparisons(df)
    all_tests.extend(tests)
    all_effects.extend(effects)

    tests_df = pd.DataFrame(all_tests)
    effects_df = pd.DataFrame(all_effects)

    return tests_df, effects_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the complete statistical-analysis pipeline."""

    print("Fairness-Aware HPO Benchmark — Statistical Analysis")
    print("=" * 60)

    try:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        df = load_and_validate()

        tests, effects = run_analysis(df)

        # Save machine-readable outputs.
        tests.to_csv(
            STAT_TESTS,
            index=False,
        )

        effects.to_csv(
            EFFECT_SIZES,
            index=False,
        )

        create_markdown_summary(
            df,
            tests,
            effects,
        )

        # Final report.
        print()
        print("[OK] Statistical analysis completed.")
        print()
        print(f"Rows analysed          : {len(df)}")
        print(
            "Overall model pairs    : "
            f"{len(DATASETS) * len(OPTIMIZERS) * len(FOLDS)} per metric"
        )
        print(
            "Overall optimizer pairs: "
            f"{len(DATASETS) * len(MODELS) * len(FOLDS)} per metric"
        )
        print("Fairness dataset       : adult (sensitive attribute: sex)")
        print()
        print("Generated files:")
        print(f"  {STAT_TESTS}")
        print(f"  {EFFECT_SIZES}")
        print(f"  {SUMMARY_MD}")
        print()

        return 0

    except SystemExit:
        raise

    except Exception as exc:
        print()
        print("[ERROR] Statistical analysis failed.")
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())