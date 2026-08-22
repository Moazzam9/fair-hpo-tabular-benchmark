# Fair HPO Tabular Benchmark

**A reproducible empirical benchmark of conventional and metaheuristic hyperparameter optimization methods for tabular classification.**

This project investigates the comparative performance, computational cost, convergence behavior, fairness characteristics, and statistical robustness of hyperparameter optimization (HPO) strategies under a controlled experimental framework.

The completed benchmark currently evaluates Random Search and Bayesian Optimization using Random Forest and XGBoost across three tabular classification datasets.

---

## Research Objective

The primary objective is to determine how different HPO strategies compare when evaluated under a consistent and reproducible protocol.

The broader research design considers:

* Default hyperparameters
* Grid Search
* Random Search
* Bayesian Optimization
* Genetic Algorithm (GA)
* Particle Swarm Optimization (PSO)
* Differential Evolution (DE)

The currently completed benchmark analysis focuses on:

* Random Search
* Bayesian Optimization

The benchmark uses three tabular classification datasets:

* Breast Cancer Wisconsin Diagnostic (WDBC)
* Adult
* Bank Marketing

and two classification models:

* Random Forest
* XGBoost

The purpose is to provide controlled empirical evidence rather than assume in advance that any particular optimizer or model is universally superior.

---

## Experimental Design

The benchmark is designed around controlled and reproducible evaluation.

The experimental protocol includes:

* Nested cross-validation
* Fixed outer folds
* Fixed inner folds
* Equal objective-evaluation budgets for stochastic optimizers
* Controlled random seeds
* Primary and secondary classification metrics
* Fairness metrics where sensitive attributes are configured
* Runtime measurements
* Convergence analysis
* Statistical analysis

The use of nested cross-validation is intended to separate hyperparameter selection from final model evaluation and reduce optimistic performance estimates.

Equal evaluation budgets are used for stochastic optimization methods to support a fair comparison of optimization efficiency rather than simply comparing methods with unequal computational opportunities.

---

## Completed Benchmark Matrix

The currently completed benchmark evaluates:

| Component | Configurations |
| --- | --- |
| Datasets | WDBC, Adult, Bank Marketing |
| Models | Random Forest, XGBoost |
| HPO methods | Random Search, Bayesian Optimization |
| Outer folds | 5 |
| Total fold-level experiments | 60 |
| Experiment groups | 12 |
| Predictive metrics | Accuracy, Balanced Accuracy, F1, ROC AUC |
| Fairness metrics | Demographic Parity Difference, Equal Opportunity Difference |
| Efficiency metric | Runtime |

The completed result set contains:

**3 datasets × 2 models × 2 optimizers × 5 outer folds = 60 experiments.**

The 60 records form 12 experimental groups:

**3 datasets × 2 models × 2 optimizers = 12 groups.**

Each experimental group contains five outer-fold results.

---

## Evaluation Framework

The benchmark evaluates HPO methods from multiple perspectives.

### Predictive Performance

The following classification metrics are evaluated:

* Accuracy
* Balanced Accuracy
* F1
* ROC AUC

These metrics provide complementary views of predictive performance, particularly where class imbalance may make raw accuracy insufficient.

### Fairness

Where a sensitive attribute is configured, the benchmark evaluates:

* Demographic Parity Difference
* Equal Opportunity Difference

The current fairness-applicable dataset is **Adult**, using `sex` as the configured sensitive attribute.

WDBC and Bank Marketing currently have no configured sensitive attributes.

Their recorded `0.0` fairness values therefore **must not be interpreted as evidence of absence of bias**. For those datasets, fairness metrics are treated as not applicable because no sensitive attribute is configured for evaluation.

### Computational Cost

Runtime is recorded in seconds for each outer-fold experiment.

Runtime distributions are analyzed using summary statistics and an interquartile-range (IQR) outlier rule:

```text
IQR = Q3 - Q1
Upper outlier threshold = Q3 + 1.5 × IQR