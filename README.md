# Fair HPO Tabular Benchmark

**An Empirical Audit of Performance, Fairness, and Compute Cost in Tabular Hyperparameter Optimization**

**Author:** Moazzam Azam — Independent Researcher — moazzamkk13@gmail.com

---

## Overview

This repository contains the full code, results, and manuscript for a controlled empirical audit comparing **Random Search** and **Bayesian Optimization** applied to **Random Forest** and **XGBoost** classifiers across three tabular classification datasets.

The audit evaluates three dimensions that are rarely combined in HPO literature:

- **Predictive Performance** — Accuracy, Balanced Accuracy, F1-score, ROC AUC
- **Group Fairness** — Demographic Parity Difference (DPD), Equal Opportunity Difference (EOD)
- **Computational Cost** — Runtime distributions and outlier analysis

---

## Paper / Manuscript

The full research paper is available in the `docs/` folder:

| Format | File |
|---|---|
| Markdown | [`docs/manuscript_draft.md`](docs/manuscript_draft.md) |
| LaTeX (for PDF) | [`docs/manuscript_draft.tex`](docs/manuscript_draft.tex) |
| Bibliography | [`docs/manuscript.bib`](docs/manuscript.bib) |
| Figures | [`docs/figures/`](docs/figures/) |

> To compile the PDF: upload the `docs/` folder contents to [Overleaf](https://www.overleaf.com) and click **Compile**.

---

## Key Findings

- **XGBoost outperforms Random Forest** across all predictive metrics (balanced accuracy 0.8186 vs. 0.7973, p < 0.001, Cohen's d_z = 1.41 — large effect)
- **Bayesian HPO** provides only marginal gains over Random Search (balanced accuracy +0.0038, small effect)
- **Fairness disparities persist** on the Adult dataset regardless of optimizer choice (DPD ≈ 0.16–0.18, p > 0.05 for optimizer comparison)
- **Runtime anomaly identified**: A single XGBoost Bayesian fold took 14.44 hours (140× the group median), attributed to a system-level execution artifact, not hyperparameter geometry

---

## Benchmark Matrix

| Component | Detail |
|---|---|
| Datasets | WDBC, Adult, Bank Marketing |
| Models | Random Forest, XGBoost |
| HPO Methods | Random Search, Bayesian Optimization |
| Outer Folds | 5-fold nested cross-validation |
| Total Experiments | 60 (3 datasets × 2 models × 2 optimizers × 5 folds) |
| Fairness Dataset | Adult (`sex` attribute: Male/Female) |

---

## Repository Structure

```
fair-hpo-tabular-benchmark/
├── docs/
│   ├── manuscript_draft.md      # Paper (Markdown)
│   ├── manuscript_draft.tex     # Paper (LaTeX)
│   ├── manuscript.bib           # BibTeX bibliography
│   ├── figures/                 # Publication-ready figures
│   └── dataset_registry.md     # Dataset DOIs and licenses
├── results/
│   └── publication/             # Final CSVs: metrics, stats, outliers
├── src/                         # Benchmark source code
├── FINAL_AUDIT.md              # Final audit summary report
└── README.md
```

---

## Experimental Design

- **Nested cross-validation**: 5-fold outer / 3-fold inner loop
- **Equal budget**: 20 iterations per HPO run to ensure fair comparison
- **Objective**: Balanced Accuracy (inner loop)
- **Seeds**: Fixed for reproducibility
- **Statistical tests**: Paired Wilcoxon signed-rank tests with Cohen's d_z effect sizes

---

## Research Objective

The primary goal is to provide rigorous, multi-dimensional empirical evidence on how HPO strategy and model choice jointly affect performance, fairness, and runtime — rather than assuming any optimizer is universally superior.

### Planned Extensions

- Metaheuristic optimizers: **Genetic Algorithm (GA)**, **Particle Swarm Optimization (PSO)**, **Differential Evolution (DE)**
- **Fairness-constrained HPO**: optimizing jointly for performance and group fairness
- Full convergence trajectory logging

---

## Datasets

All datasets are sourced from the UCI Machine Learning Repository:

| Dataset | Instances | Features | Task |
|---|---|---|---|
| [WDBC](https://doi.org/10.24432/C5DW2B) | 569 | 30 | Cancer diagnosis |
| [Adult](https://doi.org/10.24432/C5XW20) | 48,842 | 14 | Income prediction |
| [Bank Marketing](https://doi.org/10.24432/C5K306) | 45,211 | 16 | Deposit subscription |

---

## Citation

If this work is useful, please cite:

```bibtex
@misc{azam2026hpofair,
  author    = {Moazzam Azam},
  title     = {An Empirical Audit of Performance, Fairness, and Compute Cost in Tabular Hyperparameter Optimization},
  year      = {2026},
  url       = {https://github.com/Moazzam9/fair-hpo-tabular-benchmark}
}
```