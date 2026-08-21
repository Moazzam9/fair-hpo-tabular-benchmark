# Fair HPO Tabular Benchmark

**A reproducible empirical benchmark of conventional and metaheuristic hyperparameter optimization methods for tabular classification.**

This project investigates the comparative performance, computational cost, convergence behavior, and statistical robustness of commonly used hyperparameter optimization (HPO) strategies under a controlled experimental framework.

## Research Objective

The primary objective is to determine how conventional and metaheuristic HPO methods compare when evaluated under a consistent and reproducible protocol.

The benchmark compares:

* Default hyperparameters
* Grid Search
* Random Search
* Bayesian Optimization
* Genetic Algorithm (GA)
* Particle Swarm Optimization (PSO)
* Differential Evolution (DE)

across three tabular classification datasets:

* Breast Cancer Wisconsin Diagnostic (WDBC)
* Adult
* Bank Marketing

and two classification models:

* Random Forest
* XGBoost

---

## Experimental Design

The benchmark is designed around controlled and reproducible evaluation.

The experimental protocol includes:

* Nested cross-validation
* Fixed outer folds
* Fixed inner folds
* Equal objective-evaluation budgets for stochastic optimizers
* Multiple independent random seeds
* Primary and secondary classification metrics
* Runtime measurements
* Convergence analysis
* Statistical significance testing
* Effect-size analysis

The use of nested cross-validation is intended to separate hyperparameter selection from final model evaluation and reduce optimistic performance estimates.

Equal evaluation budgets are used for stochastic optimization methods to support a fair comparison of optimization efficiency rather than simply comparing methods with unequal computational opportunities.

---

## Benchmark Matrix

The core benchmark consists of the following combinations:

| Component             | Configurations                               |
| --------------------- | -------------------------------------------- |
| Datasets              | WDBC, Adult, Bank Marketing                  |
| Models                | Random Forest, XGBoost                       |
| HPO methods           | Default, Grid, Random, Bayesian, GA, PSO, DE |
| Evaluation            | Nested cross-validation                      |
| Stochastic evaluation | Multiple independent seeds                   |
| Efficiency analysis   | Runtime and objective evaluations            |
| Optimization analysis | Convergence                                  |
| Statistical analysis  | Significance tests and effect sizes          |

This produces a controlled comparison across dataset, model, and optimization strategy.

---

## Evaluation Framework

The benchmark evaluates HPO methods from multiple perspectives.

### Predictive Performance

Primary and secondary classification metrics are collected to evaluate predictive performance across datasets and models.

### Computational Cost

Runtime measurements and objective-evaluation counts are recorded to assess the computational efficiency of each optimization strategy.

### Convergence

Optimization trajectories are analyzed to investigate how quickly different methods improve their objective values within the available evaluation budget.

### Statistical Analysis

The study includes statistical significance testing and effect-size analysis to distinguish potentially meaningful differences from differences attributable to experimental variability.

---

## Reproducibility

Reproducibility is a core requirement of the benchmark.

The experimental design specifies:

* Fixed cross-validation folds
* Fixed evaluation protocols
* Controlled random seeds
* Controlled optimization budgets
* Consistent model evaluation
* Consistent preprocessing
* Repeated independent runs

The objective is to ensure that comparisons between HPO strategies are attributable to the optimization method rather than uncontrolled differences in the experimental procedure.

---

## Datasets

The benchmark uses:

1. **Breast Cancer Wisconsin Diagnostic (WDBC)**
2. **Adult**
3. **Bank Marketing**

Datasets are obtained from authoritative sources within the **UCI Machine Learning Repository**.

Raw datasets are **not committed to this repository**.

Dataset acquisition and preprocessing are handled as part of the experimental pipeline.

---

## Project Status

The project is organized into five stages:

| Stage   | Description              |
| ------- | ------------------------ |
| Stage 1 | Experimental design      |
| Stage 2 | Implementation and pilot |
| Stage 3 | Full experiments         |
| Stage 4 | Statistical analysis     |
| Stage 5 | Thesis / paper reporting |

**Current project status:** Experimental design and implementation/pilot phase.

Results, statistical conclusions, and final research claims will be added only after the corresponding experimental stages have been completed.

---

## Research Scope

The current benchmark is intentionally limited to:

* Tabular classification
* Binary classification datasets
* Random Forest
* XGBoost
* Seven HPO configurations
* Three benchmark datasets

The purpose is to provide a controlled empirical comparison rather than to claim universal superiority of any particular optimization method.

---

## Intended Outputs

The completed benchmark is intended to produce:

* Reproducible experimental results
* Model performance comparisons
* HPO performance comparisons
* Runtime comparisons
* Convergence analyses
* Statistical significance analyses
* Effect-size analyses
* Publication-quality tables and figures
* Thesis/paper-ready empirical evidence

No conclusions about optimizer superiority are assumed in advance; conclusions will be based on the results produced by the experimental protocol.

---

## Repository Principle

> **Controlled experiments, equal evaluation budgets, repeated runs, statistical evidence, and reproducible conclusions.**

---

## License

See [`LICENSE`](LICENSE) for licensing information.

## Citation

Citation information will be provided in [`CITATION.cff`](CITATION.cff).
