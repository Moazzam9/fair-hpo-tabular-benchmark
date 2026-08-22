# Publication Summary — Fairness-Aware HPO Benchmark

## Overview

This report summarizes the completed fairness-aware hyperparameter optimization benchmark across:

- **Datasets:** Adult, Bank Marketing, WDBC
- **Models:** Random Forest, XGBoost
- **Optimizers:** Random Search, Bayesian Optimization
- **Outer folds:** 5
- **Total experiments:** 60
- **Experiment groups:** 12

The analysis is based exclusively on the existing benchmark outputs and does not perform additional model training, hyperparameter optimization, or data downloading.

## Data and Fairness Scope

The benchmark contains 60 fold-level observations covering all combinations of the three datasets, two models, two optimizers, and five outer folds.

Fairness metrics are applicable only to **Adult**, where the configured sensitive attribute is `sex`.

Bank Marketing and WDBC currently have no configured sensitive attributes. Their recorded `0.0` fairness values therefore indicate that fairness was not evaluated for a configured sensitive attribute; they **must not be interpreted as evidence of absence of bias**.

## Performance Findings

### Adult

XGBoost outperformed Random Forest across the primary predictive metrics:

- Accuracy: approximately **0.872 vs 0.859**
- Balanced accuracy: approximately **0.790 vs 0.769**
- F1: approximately **0.703 vs 0.669**
- ROC AUC: approximately **0.927 vs 0.910**

Bayesian optimization also produced slightly higher average predictive performance than random search.

### Bank Marketing

XGBoost showed modestly higher performance than Random Forest:

- Accuracy: approximately **0.906 vs 0.905**
- Balanced accuracy: approximately **0.706 vs 0.676**
- F1: approximately **0.526 vs 0.481**
- ROC AUC: approximately **0.927 vs 0.922**

The optimizer effect was comparatively small for accuracy and ROC AUC, while Bayesian optimization improved balanced accuracy and F1 modestly.

### WDBC

XGBoost achieved higher mean accuracy, balanced accuracy, F1, and ROC AUC than Random Forest, although the differences were relatively small.

Random Search slightly exceeded Bayesian optimization on mean accuracy, balanced accuracy, and F1 for WDBC, while the ROC AUC difference was very small.

## Fairness Findings

For Adult, fairness metrics were computed against the configured `sex` attribute.

Mean demographic parity difference was approximately:

- Random Forest + Random Search: **0.177**
- Random Forest + Bayesian: **0.160**
- XGBoost + Random Search: **0.167**
- XGBoost + Bayesian: **0.173**

Mean equal opportunity difference was approximately:

- Random Forest + Random Search: **0.083**
- Random Forest + Bayesian: **0.086**
- XGBoost + Random Search: **0.078**
- XGBoost + Bayesian: **0.070**

These values indicate measurable group disparities in the Adult benchmark and should be interpreted in the context of the selected fairness definitions and sensitive attribute.

No comparable fairness conclusion should be drawn for WDBC or Bank Marketing because no sensitive attribute is currently configured for those datasets.

## Runtime Findings

Runtime varied substantially between experimental groups.

The most important observation is:

**Adult / XGBoost / Bayesian / Fold 5: 51,992.587 seconds (~14.44 hours).**

This observation exceeds the group-level IQR upper bound and is classified as an extreme runtime outlier.

The other detected runtime outliers were:

- Adult / Random Forest / Bayesian / Fold 2: ~596.8 s
- Bank Marketing / Random Forest / Random / Fold 3: ~827.3 s
- Bank Marketing / XGBoost / Bayesian / Fold 2: ~352.5 s
- WDBC / XGBoost / Random / Fold 1: ~27.2 s

The IQR rule used for detection was:

`upper bound = Q3 + 1.5 × IQR`

Because of the Adult/XGBoost/Bayesian extreme observation, mean runtime is substantially larger than median runtime for that group. Runtime comparisons should therefore consider both statistics.

## Model Comparison

Across the benchmark, XGBoost generally achieved higher predictive performance than Random Forest.

The clearest improvements occurred for:

- Adult balanced accuracy
- Adult F1
- Bank Marketing balanced accuracy
- Bank Marketing F1

On WDBC, XGBoost also generally performed better, although the differences were smaller.

Runtime behavior depended strongly on dataset and optimizer. XGBoost was substantially faster than Random Forest on Bank Marketing and WDBC, while the Adult Bayesian configuration was dominated by the single extreme Fold-5 observation.

## Optimizer Comparison

Bayesian optimization generally produced small predictive-performance improvements on Adult and Bank Marketing.

For WDBC, Random Search slightly outperformed Bayesian optimization on several predictive metrics.

Runtime differences between optimizers were generally modest except for Adult, where Bayesian optimization was strongly affected by the 51,992.587-second XGBoost Fold-5 observation.

Consequently, the Adult Bayesian mean runtime should not be interpreted as representative of the typical runtime of that configuration; its median is much more representative of the five-fold distribution.

## Anomaly Interpretation

The runtime anomalies were identified statistically rather than manually selected.

The Adult/XGBoost/Bayesian/Fold-5 observation is especially important for reproducibility and publication reporting because it materially changes aggregate runtime statistics.

It should remain visible in the reported results rather than being silently removed. Median and distribution-aware statistics should accompany mean runtime values.

## Reproducibility

All analyses represented here were generated from the existing benchmark result files.

No additional:

- model training,
- hyperparameter optimization,
- dataset downloading, or
- modification of `benchmark_results.csv` or `benchmark_results.json`

is required to reproduce the publication-analysis tables and figures.

## Generated Artifacts

The publication-analysis directory contains:

- `summary_by_dataset_model_optimizer.csv`
- `fold_level_results.csv`
- `fairness_analysis.csv`
- `runtime_analysis.csv`
- `model_comparison.csv`
- `optimizer_comparison.csv`
- `anomaly_analysis.csv`

Figures:

- `figures/performance_comparison.png`
- `figures/fairness_comparison.png`
- `figures/runtime_comparison.png`

