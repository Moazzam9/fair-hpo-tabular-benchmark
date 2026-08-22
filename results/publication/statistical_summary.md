# Statistical Analysis — Fairness-Aware HPO Benchmark

## Overview

The benchmark contains **60 fold-level observations** covering **3 datasets × 2 models × 2 optimizers × 5 outer folds**.

All primary comparisons are paired at the outer-fold level. The statistical analysis uses paired t-tests, Wilcoxon signed-rank tests, and Cohen's dz effect sizes.

## Pairing Methodology

- **Model comparison:** Random Forest vs XGBoost, paired by dataset + optimizer + outer fold; 30 pairs per metric.
- **Optimizer comparison:** Random Search vs Bayesian Optimization, paired by dataset + model + outer fold; 30 pairs per metric.
- **Adult fairness comparisons:** paired using the corresponding experimental dimensions; 10 pairs per fairness metric.

## Primary Statistical Results

### Random Forest vs XGBoost

- **accuracy**: random_forest mean = 0.9054, xgboost mean = 0.9138, mean difference = 0.0084, Wilcoxon p = <0.001, Cohen's dz = 0.7669.
- **balanced_accuracy**: random_forest mean = 0.7973, xgboost mean = 0.8186, mean difference = 0.0214, Wilcoxon p = <0.001, Cohen's dz = 1.4124.
- **f1**: random_forest mean = 0.6949, xgboost mean = 0.7263, mean difference = 0.0314, Wilcoxon p = <0.001, Cohen's dz = 1.4774.
- **roc_auc**: random_forest mean = 0.9411, xgboost mean = 0.9491, mean difference = 0.0080, Wilcoxon p = <0.001, Cohen's dz = 0.9118.
- **runtime_seconds**: random_forest mean = 386.9720, xgboost mean = 1933.5200, mean difference = 1546.5479, Wilcoxon p = <0.001, Cohen's dz = 0.1638.

### Random Search vs Bayesian Optimization

- **accuracy**: random mean = 0.9088, bayesian mean = 0.9103, mean difference = 0.0015, Wilcoxon p = 0.0321, Cohen's dz = 0.2616.
- **balanced_accuracy**: random mean = 0.8061, bayesian mean = 0.8098, mean difference = 0.0038, Wilcoxon p = 0.0450, Cohen's dz = 0.4023.
- **f1**: random mean = 0.7073, bayesian mean = 0.7139, mean difference = 0.0066, Wilcoxon p = 0.0074, Cohen's dz = 0.5364.
- **roc_auc**: random mean = 0.9436, bayesian mean = 0.9465, mean difference = 0.0030, Wilcoxon p = 0.0201, Cohen's dz = 0.4635.
- **runtime_seconds**: random mean = 293.8349, bayesian mean = 2026.6571, mean difference = 1732.8222, Wilcoxon p = <0.001, Cohen's dz = 0.1837.

## Dataset-Specific Interpretation

### wdbc

**Model comparison:**
- accuracy: mean difference = 0.0105, Wilcoxon p = 0.1914, Cohen's dz = 0.6194 (see effect_sizes.csv).
- balanced_accuracy: mean difference = 0.0125, Wilcoxon p = 0.1738, Cohen's dz = 0.6309 (see effect_sizes.csv).
- f1: mean difference = 0.0149, Wilcoxon p = 0.1504, Cohen's dz = 0.6220 (see effect_sizes.csv).
- roc_auc: mean difference = 0.0023, Wilcoxon p = 0.1055, Cohen's dz = 0.5684 (see effect_sizes.csv).
- runtime_seconds: mean difference = -12.2590, Wilcoxon p = 0.0020, Cohen's dz = -4.1179 (see effect_sizes.csv).

**Optimizer comparison:**
- accuracy: mean difference = -0.0018, Wilcoxon p = 0.7500, Cohen's dz = -0.2535 (see effect_sizes.csv).
- balanced_accuracy: mean difference = -0.0029, Wilcoxon p = 0.4375, Cohen's dz = -0.4197 (see effect_sizes.csv).
- f1: mean difference = -0.0028, Wilcoxon p = 0.3125, Cohen's dz = -0.3005 (see effect_sizes.csv).
- roc_auc: mean difference = -0.0007, Wilcoxon p = 0.6523, Cohen's dz = -0.3161 (see effect_sizes.csv).
- runtime_seconds: mean difference = 14.2474, Wilcoxon p = 0.0020, Cohen's dz = 5.3156 (see effect_sizes.csv).

### adult

**Model comparison:**
- accuracy: mean difference = 0.0130, Wilcoxon p = 0.0020, Cohen's dz = 3.0596 (see effect_sizes.csv).
- balanced_accuracy: mean difference = 0.0212, Wilcoxon p = 0.0020, Cohen's dz = 2.5550 (see effect_sizes.csv).
- f1: mean difference = 0.0335, Wilcoxon p = 0.0020, Cohen's dz = 4.2623 (see effect_sizes.csv).
- roc_auc: mean difference = 0.0175, Wilcoxon p = 0.0020, Cohen's dz = 2.1702 (see effect_sizes.csv).
- runtime_seconds: mean difference = 4984.7555, Wilcoxon p = 0.0840, Cohen's dz = 0.3048 (see effect_sizes.csv).

**Optimizer comparison:**
- accuracy: mean difference = 0.0059, Wilcoxon p = 0.0039, Cohen's dz = 1.3347 (see effect_sizes.csv).
- balanced_accuracy: mean difference = 0.0047, Wilcoxon p = 0.2324, Cohen's dz = 0.5761 (see effect_sizes.csv).
- f1: mean difference = 0.0096, Wilcoxon p = 0.0098, Cohen's dz = 1.2648 (see effect_sizes.csv).
- roc_auc: mean difference = 0.0084, Wilcoxon p = 0.0020, Cohen's dz = 1.0689 (see effect_sizes.csv).
- runtime_seconds: mean difference = 5175.1412, Wilcoxon p = 0.1309, Cohen's dz = 0.3167 (see effect_sizes.csv).

### bank_marketing

**Model comparison:**
- accuracy: mean difference = 0.0017, Wilcoxon p = 0.0371, Cohen's dz = 0.8315 (see effect_sizes.csv).
- balanced_accuracy: mean difference = 0.0304, Wilcoxon p = 0.0020, Cohen's dz = 3.0784 (see effect_sizes.csv).
- f1: mean difference = 0.0458, Wilcoxon p = 0.0020, Cohen's dz = 2.7232 (see effect_sizes.csv).
- roc_auc: mean difference = 0.0042, Wilcoxon p = 0.0098, Cohen's dz = 1.1342 (see effect_sizes.csv).
- runtime_seconds: mean difference = -332.8526, Wilcoxon p = 0.0020, Cohen's dz = -3.1017 (see effect_sizes.csv).

**Optimizer comparison:**
- accuracy: mean difference = 0.0003, Wilcoxon p = 0.5566, Cohen's dz = 0.2535 (see effect_sizes.csv).
- balanced_accuracy: mean difference = 0.0094, Wilcoxon p = 0.0098, Cohen's dz = 1.0413 (see effect_sizes.csv).
- f1: mean difference = 0.0131, Wilcoxon p = 0.0137, Cohen's dz = 0.9381 (see effect_sizes.csv).
- roc_auc: mean difference = 0.0012, Wilcoxon p = 0.3750, Cohen's dz = 0.3292 (see effect_sizes.csv).
- runtime_seconds: mean difference = 9.0780, Wilcoxon p = 0.3223, Cohen's dz = 0.0918 (see effect_sizes.csv).

## Fairness Analysis

Fairness inference is restricted to **Adult**, where the configured sensitive attribute is `sex`.

WDBC and Bank Marketing are excluded from inferential fairness comparisons because no sensitive attribute is currently configured.

- **fairness_model / demographic_parity_difference**: mean difference = 0.0017, Wilcoxon p = 0.4922, Cohen's dz = 0.1415.
- **fairness_optimizer / demographic_parity_difference**: mean difference = -0.0055, Wilcoxon p = 0.2324, Cohen's dz = -0.4466.
- **fairness_model / equal_opportunity_difference**: mean difference = -0.0100, Wilcoxon p = 0.1934, Cohen's dz = -0.4546.
- **fairness_optimizer / equal_opportunity_difference**: mean difference = -0.0026, Wilcoxon p = 0.6250, Cohen's dz = -0.1727.

## Interpretation Guidelines

A p-value below 0.05 is treated as evidence against the null hypothesis for the corresponding paired test, but statistical significance should not be interpreted as practical importance without considering the effect size and magnitude of the observed difference.

Cohen's dz is interpreted using conventional descriptive thresholds: <0.2 negligible, 0.2–<0.5 small, 0.5–<0.8 medium, and ≥0.8 large.

No multiple-comparison correction is applied automatically. The resulting p-values should therefore be treated as exploratory unless a prespecified multiplicity procedure is adopted for the final publication.

## Reproducibility

This analysis reads the existing `results/benchmark_results.csv` file only. It does not perform model training, HPO, or dataset downloading and does not modify the benchmark result files.
