"""
Publication Analysis & Summary Generator for HPO & Fairness Benchmark.

This script processes pre-computed benchmark results from `results/benchmark_results.csv`,
validates experiment structure, performs statistical analyses across performance,
fairness, and runtime dimensions, identifies runtime anomalies using an IQR rule,
and generates publication-ready CSV summary tables and 300 DPI figures.

Methodology & Design Decisions:
1. Validation: Ensures exact row count (60 rows = 3 datasets * 2 models * 2 optimizers * 5 folds)
   and expected dimension values. Exits with non-zero status if validation fails.
2. Fairness Context: Explicitly distinguishes datasets with configured sensitive attributes
   ('adult') from those without ('wdbc', 'bank_marketing'). 0.0 values in unconfigured datasets
   represent lack of sensitive attribute tracking rather than proof of absence of bias.
3. Outlier Detection: Applies the Interquartile Range (IQR) rule (Q3 + 1.5 * IQR) to detect
   runtime anomalies, specifically highlighting extreme observations like Adult/XGBoost/Bayesian/Fold-5.
4. Summary Statistics: Computes count, mean, std, median, min, max for all key metrics.
5. Visualization: Generates 300 DPI PNG plots using matplotlib and seaborn with clear labels.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set random seed and plot style for determinism
np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#eeeeee",
    "grid.linestyle": "--",
    "figure.autolayout": True,
})


def validate_benchmark_data(df: pd.DataFrame) -> None:
    """
    Validate that the benchmark results dataframe matches the expected experimental schema.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded benchmark dataset.

    Raises
    ------
    SystemExit
        Exits with non-zero status if any validation check fails.
    """
    expected_rows = 60
    if len(df) != expected_rows:
        print(f"Validation Error: Expected exactly {expected_rows} rows, found {len(df)}.", file=sys.stderr)
        sys.exit(1)

    expected_datasets = {"wdbc", "adult", "bank_marketing"}
    expected_models = {"random_forest", "xgboost"}
    expected_optimizers = {"random", "bayesian"}
    expected_folds = {1, 2, 3, 4, 5}

    actual_datasets = set(df["dataset"].unique())
    actual_models = set(df["model"].unique())
    actual_optimizers = set(df["optimizer"].unique())
    actual_folds = set(df["outer_fold"].unique())

    errors = []
    if actual_datasets != expected_datasets:
        errors.append(f"Datasets mismatch: expected {expected_datasets}, got {actual_datasets}")
    if actual_models != expected_models:
        errors.append(f"Models mismatch: expected {expected_models}, got {actual_models}")
    if actual_optimizers != expected_optimizers:
        errors.append(f"Optimizers mismatch: expected {expected_optimizers}, got {actual_optimizers}")
    if actual_folds != expected_folds:
        errors.append(f"Folds mismatch: expected {expected_folds}, got {actual_folds}")

    if errors:
        for err in errors:
            print(f"Validation Error: {err}", file=sys.stderr)
        sys.exit(1)

    # Check for grid completeness (no missing combinations)
    grouped = df.groupby(["dataset", "model", "optimizer", "outer_fold"]).size()
    if len(grouped) != expected_rows or (grouped != 1).any():
        print("Validation Error: Experiment grid is incomplete or contains duplicate fold records.", file=sys.stderr)
        sys.exit(1)

    print("[OK] Dataset validation passed: 60 rows matching 3 datasets x 2 models x 2 optimizers x 5 folds.")


def generate_fold_level_results(df: pd.DataFrame, pub_dir: Path) -> pd.DataFrame:
    """
    Export detailed fold-level results with fairness applicability indicators.
    """
    fold_df = df.copy()
    fold_df["fairness_applicable"] = fold_df["dataset"] == "adult"
    out_path = pub_dir / "fold_level_results.csv"
    fold_df.to_csv(out_path, index=False)
    return fold_df


def generate_summary_by_dataset_model_optimizer(df: pd.DataFrame, pub_dir: Path) -> pd.DataFrame:
    """
    Compute summary statistics (count, mean, std, median, min, max) grouped by dataset, model, and optimizer.
    """
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "demographic_parity_difference",
        "equal_opportunity_difference",
        "runtime_seconds",
    ]

    agg_funcs = ["count", "mean", "std", "median", "min", "max"]

    grouped = df.groupby(["dataset", "model", "optimizer"])[metrics].agg(agg_funcs)

    # Flatten column MultiIndex
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()

    out_path = pub_dir / "summary_by_dataset_model_optimizer.csv"
    grouped.to_csv(out_path, index=False)
    return grouped


def generate_fairness_analysis(df: pd.DataFrame, pub_dir: Path) -> pd.DataFrame:
    """
    Generate detailed fairness analysis, explicitly distinguishing datasets with configured
    sensitive attributes (adult) from unconfigured ones (wdbc, bank_marketing).
    """
    fairness_metrics = ["demographic_parity_difference", "equal_opportunity_difference"]
    agg_funcs = ["count", "mean", "std", "median", "min", "max"]

    records = []
    for (dataset, model, optimizer), group in df.groupby(["dataset", "model", "optimizer"]):
        is_adult = dataset == "adult"
        status = "Configured (Sensitive Attribute: sex)" if is_adult else "Not Configured (Sensitive Attribute: None)"
        note = (
            "Valid fairness metric computed against sensitive attribute 'sex'."
            if is_adult
            else "Metric value 0.0 reflects lack of sensitive attribute tracking, NOT proof of absence of bias."
        )

        row = {
            "dataset": dataset,
            "model": model,
            "optimizer": optimizer,
            "fairness_applicable": is_adult,
            "fairness_status": status,
            "methodology_note": note,
        }

        for metric in fairness_metrics:
            if is_adult:
                for stat in agg_funcs:
                    val = getattr(group[metric], stat)()
                    row[f"{metric}_{stat}"] = val
            else:
                for stat in agg_funcs:
                    row[f"{metric}_{stat}"] = np.nan if stat != "count" else len(group)

        records.append(row)

    fairness_df = pd.DataFrame(records)
    out_path = pub_dir / "fairness_analysis.csv"
    fairness_df.to_csv(out_path, index=False)
    return fairness_df


def generate_runtime_and_anomaly_analysis(df: pd.DataFrame, pub_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform runtime analysis and identify extreme observations using an IQR outlier rule.

    IQR Rule: Outlier if runtime > Q3 + 1.5 * IQR within group or overall dataset.
    """
    runtime_records = []
    anomaly_records = []

    # Group-level analysis
    for (dataset, model, optimizer), group in df.groupby(["dataset", "model", "optimizer"]):
        runtimes = group["runtime_seconds"]
        q1 = runtimes.quantile(0.25)
        q3 = runtimes.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        outliers = group[group["runtime_seconds"] > upper_bound]

        runtime_records.append({
            "dataset": dataset,
            "model": model,
            "optimizer": optimizer,
            "count": len(runtimes),
            "mean_seconds": runtimes.mean(),
            "std_seconds": runtimes.std(),
            "median_seconds": runtimes.median(),
            "min_seconds": runtimes.min(),
            "max_seconds": runtimes.max(),
            "q1_seconds": q1,
            "q3_seconds": q3,
            "iqr_seconds": iqr,
            "upper_bound_seconds": upper_bound,
            "outlier_count": len(outliers),
        })

        for _, row in outliers.iterrows():
            is_extreme_target = (
                dataset == "adult"
                and model == "xgboost"
                and optimizer == "bayesian"
                and row["outer_fold"] == 5
            )

            anomaly_records.append({
                "dataset": dataset,
                "outer_fold": row["outer_fold"],
                "model": model,
                "optimizer": optimizer,
                "runtime_seconds": row["runtime_seconds"],
                "group_q3": q3,
                "group_iqr": iqr,
                "iqr_upper_bound": upper_bound,
                "exceeds_upper_bound_ratio": row["runtime_seconds"] / upper_bound if upper_bound > 0 else np.nan,
                "anomaly_category": "Extreme Runtime Outlier" if is_extreme_target else "Runtime Outlier",
                "is_adult_xgboost_bayesian_fold5": is_extreme_target,
                "description": (
                    f"Extreme runtime observation of {row['runtime_seconds']:.3f}s (~{row['runtime_seconds']/3600:.2f}h) "
                    f"substantially exceeding IQR threshold ({upper_bound:.3f}s)."
                ) if is_extreme_target else f"Runtime of {row['runtime_seconds']:.3f}s exceeds IQR upper bound ({upper_bound:.3f}s)."
            })

    runtime_df = pd.DataFrame(runtime_records)
    out_path_runtime = pub_dir / "runtime_analysis.csv"
    runtime_df.to_csv(out_path_runtime, index=False)

    anomaly_df = pd.DataFrame(anomaly_records)
    out_path_anomaly = pub_dir / "anomaly_analysis.csv"
    anomaly_df.to_csv(out_path_anomaly, index=False)

    return runtime_df, anomaly_df


def generate_model_comparison(df: pd.DataFrame, pub_dir: Path) -> pd.DataFrame:
    """
    Compare performance and runtime between Random Forest and XGBoost.
    """
    metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "runtime_seconds"]

    records = []
    for dataset, group in df.groupby("dataset"):
        rf_group = group[group["model"] == "random_forest"]
        xgb_group = group[group["model"] == "xgboost"]

        for metric in metrics:
            rf_mean = rf_group[metric].mean()
            xgb_mean = xgb_group[metric].mean()
            rf_std = rf_group[metric].std()
            xgb_std = xgb_group[metric].std()
            rf_median = rf_group[metric].median()
            xgb_median = xgb_group[metric].median()

            records.append({
                "dataset": dataset,
                "metric": metric,
                "random_forest_mean": rf_mean,
                "random_forest_std": rf_std,
                "random_forest_median": rf_median,
                "xgboost_mean": xgb_mean,
                "xgboost_std": xgb_std,
                "xgboost_median": xgb_median,
                "diff_mean_xgb_minus_rf": xgb_mean - rf_mean,
                "diff_median_xgb_minus_rf": xgb_median - rf_median,
            })

    comp_df = pd.DataFrame(records)
    out_path = pub_dir / "model_comparison.csv"
    comp_df.to_csv(out_path, index=False)
    return comp_df


def generate_optimizer_comparison(df: pd.DataFrame, pub_dir: Path) -> pd.DataFrame:
    """
    Compare performance and runtime between Random Search and Bayesian Optimization.
    """
    metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "runtime_seconds"]

    records = []
    for dataset, group in df.groupby("dataset"):
        rand_group = group[group["optimizer"] == "random"]
        bayes_group = group[group["optimizer"] == "bayesian"]

        for metric in metrics:
            rand_mean = rand_group[metric].mean()
            bayes_mean = bayes_group[metric].mean()
            rand_std = rand_group[metric].std()
            bayes_std = bayes_group[metric].std()
            rand_median = rand_group[metric].median()
            bayes_median = bayes_group[metric].median()

            records.append({
                "dataset": dataset,
                "metric": metric,
                "random_mean": rand_mean,
                "random_std": rand_std,
                "random_median": rand_median,
                "bayesian_mean": bayes_mean,
                "bayesian_std": bayes_std,
                "bayesian_median": bayes_median,
                "diff_mean_bayes_minus_random": bayes_mean - rand_mean,
                "diff_median_bayes_minus_random": bayes_median - rand_median,
            })

    comp_df = pd.DataFrame(records)
    out_path = pub_dir / "optimizer_comparison.csv"
    comp_df.to_csv(out_path, index=False)
    return comp_df


def generate_publication_figures(df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """
    Generate publication-quality PNG figures at 300 DPI.

    Figures generated:
    1. performance_comparison.png
    2. fairness_comparison.png
    3. runtime_comparison.png
    """
    generated_figures = []

    # Palette setup
    model_palette = {"random_forest": "#2b5c8f", "xgboost": "#d95f02"}
    opt_palette = {"random": "#7570b3", "bayesian": "#1b9e77"}

    # ---------------------------------------------------------
    # Figure 1: Performance Comparison
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    perf_metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("f1", "F1 Score"),
        ("roc_auc", "ROC AUC"),
    ]

    for idx, (metric, title) in enumerate(perf_metrics):
        ax = axes[idx // 2, idx % 2]
        sns.barplot(
            data=df,
            x="dataset",
            y=metric,
            hue="model",
            palette=model_palette,
            ax=ax,
            errorbar="sd",
            capsize=0.1,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(f"Model Comparison: {title}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Model", frameon=True)

    plt.suptitle("Benchmark Performance Overview across Datasets and Models", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig1_path = figures_dir / "performance_comparison.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures.append(fig1_path)

    # ---------------------------------------------------------
    # Figure 2: Fairness Comparison
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    adult_df = df[df["dataset"] == "adult"].copy()

    sns.barplot(
        data=adult_df,
        x="model",
        y="demographic_parity_difference",
        hue="optimizer",
        palette=opt_palette,
        ax=ax1,
        errorbar="sd",
        capsize=0.1,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_title("Demographic Parity Difference (Adult)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("Demographic Parity Difference", fontsize=11)
    ax1.legend(title="Optimizer", frameon=True)

    sns.barplot(
        data=adult_df,
        x="model",
        y="equal_opportunity_difference",
        hue="optimizer",
        palette=opt_palette,
        ax=ax2,
        errorbar="sd",
        capsize=0.1,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_title("Equal Opportunity Difference (Adult)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("Equal Opportunity Difference", fontsize=11)
    ax2.legend(title="Optimizer", frameon=True)

    # Add explanatory note for non-applicable datasets
    plt.figtext(
        0.5,
        0.01,
        "Note: Adult dataset is the only dataset with a configured sensitive attribute ('sex').\n"
        "WDBC and Bank Marketing datasets do not track sensitive attributes (fairness metrics = 0.0 by default, representing non-applicability).",
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe6", edgecolor="#ffe58f"),
    )

    plt.suptitle("Fairness Metrics Analysis (Applicable Dataset: Adult)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig2_path = figures_dir / "fairness_comparison.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures.append(fig2_path)

    # ---------------------------------------------------------
    # Figure 3: Runtime Comparison & Outlier Callout
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Log scale boxplot across groups
    df["group_label"] = df["dataset"] + "\n" + df["model"] + "\n(" + df["optimizer"] + ")"
    sns.boxplot(
        data=df,
        x="group_label",
        y="runtime_seconds",
        hue="optimizer",
        palette=opt_palette,
        ax=ax1,
        dodge=False,
    )
    ax1.set_yscale("log")
    ax1.set_title("Runtime Distribution by Group (Log Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Experimental Group", fontsize=10)
    ax1.set_ylabel("Runtime (seconds, log scale)", fontsize=11)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.legend(title="Optimizer", frameon=True)

    # Highlight extreme outlier
    sns.scatterplot(
        data=df,
        x="outer_fold",
        y="runtime_seconds",
        hue="group_label",
        ax=ax2,
        s=80,
        alpha=0.8,
    )
    ax2.set_yscale("log")
    ax2.set_title("Fold-level Runtime & Extreme Outlier Callout", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Outer Fold", fontsize=11)
    ax2.set_ylabel("Runtime (seconds, log scale)", fontsize=11)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # Annotate Adult/XGBoost/Bayesian/Fold-5 outlier
    extreme_row = df[
        (df["dataset"] == "adult")
        & (df["model"] == "xgboost")
        & (df["optimizer"] == "bayesian")
        & (df["outer_fold"] == 5)
    ].iloc[0]

    ax2.annotate(
        f"Extreme Outlier:\nAdult XGBoost Bayesian Fold-5\n({extreme_row['runtime_seconds']:.1f}s / ~14.4h)",
        xy=(5, extreme_row["runtime_seconds"]),
        xytext=(3.2, extreme_row["runtime_seconds"] / 5),
        arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=8),
        fontsize=9,
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", edgecolor="red"),
    )

    plt.suptitle("Runtime Performance & Outlier Analysis", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig3_path = figures_dir / "runtime_comparison.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures.append(fig3_path)

    return generated_figures


def main() -> None:
    """
    Main entry point for generating publication analysis tables and figures.
    """
    project_root = Path(__file__).resolve().parent.parent
    results_file = project_root / "results" / "benchmark_results.csv"

    if not results_file.exists():
        print(f"Error: Benchmark results file not found at {results_file}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Read results without mutating original CSV/JSON
    df = pd.read_csv(results_file)

    # Step 2: Validate experiment dimensions and row count
    validate_benchmark_data(df)

    # Step 3: Automatically ensure output directories exist
    pub_dir = project_root / "results" / "publication"
    figures_dir = pub_dir / "figures"
    pub_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Generate CSV artifacts
    fold_df = generate_fold_level_results(df, pub_dir)
    summary_df = generate_summary_by_dataset_model_optimizer(df, pub_dir)
    fairness_df = generate_fairness_analysis(df, pub_dir)
    runtime_df, anomaly_df = generate_runtime_and_anomaly_analysis(df, pub_dir)
    model_comp_df = generate_model_comparison(df, pub_dir)
    opt_comp_df = generate_optimizer_comparison(df, pub_dir)

    # Step 5: Generate Figures
    figure_paths = generate_publication_figures(df, figures_dir)

    # Step 6: Print Concise Final Report
    csv_files = [
        "summary_by_dataset_model_optimizer.csv",
        "fold_level_results.csv",
        "fairness_analysis.csv",
        "runtime_analysis.csv",
        "model_comparison.csv",
        "optimizer_comparison.csv",
        "anomaly_analysis.csv",
    ]

    outlier_list = [
        f"{row['dataset'].upper()}/{row['model']}/{row['optimizer']}/Fold-{row['outer_fold']} ({row['runtime_seconds']:.1f}s)"
        for _, row in anomaly_df.iterrows()
    ]

    print("\n==================================================")
    print("      PUBLICATION ANALYSIS SUMMARY REPORT         ")
    print("==================================================")
    print(f"Total Rows Validated     : {len(df)}")
    print(f"Total Experiment Groups  : {len(summary_df)}")
    print(f"Fairness-Applicable Dataset: adult (Sensitive Attribute: sex)")
    print(f"Unconfigured Fairness Datasets: wdbc, bank_marketing (Sensitive Attribute: None)")
    print(f"Runtime Outliers Detected: {len(anomaly_df)}")
    for out in outlier_list:
        print(f"  - {out}")
    print("\nGenerated CSV Files (in results/publication/):")
    for fname in csv_files:
        print(f"  - {fname}")
    print("\nGenerated PNG Figures (300 DPI, in results/publication/figures/):")
    for fig_path in figure_paths:
        print(f"  - {fig_path.name}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
