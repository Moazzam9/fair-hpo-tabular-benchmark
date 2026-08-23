# Convergence History Availability

## Status: Partially Available (Pilot Only)

### Summary

HPO iteration-level convergence histories were **generated internally** during the
benchmark but were **not persisted** for 58 of 60 experiments.

The internal optimizer functions `run_random_search()` and `run_bayesian_search()`
(in `src/fair_hpo/experiments/hpo.py`) both build a full per-iteration `history`
list and return it as part of their result dictionary. However, `run_one_experiment()`
in `scripts/run_full_benchmark.py` consumed only `outer_result`, `best_params`, and
`best_objective_score` from the returned dict. The `history` key was silently
discarded before anything was written to disk.

---

## What History Data Exists

| File | Coverage | Optimizer | Folds |
|---|---|---|---|
| `results/adult_bayesian_search.json` | Adult / Random Forest / Bayesian | Bayesian | Fold 1 only (10 iterations) |
| `results/adult_random_search.json` | Adult / Random Forest / Random | Random | Fold 1 only (10 iterations) |

These two files were saved from a **pilot run** prior to the main benchmark
and are **not** from the same execution that produced `benchmark_results.csv`.
Specifically:

- The pilot histories use a different search space (`n_estimators` up to 200) than
  the final benchmark (capped at 100), confirming they are from an earlier stage.
- They cover only 1 of 5 outer folds for 1 of 3 datasets and 1 of 2 model types.
- Their outer-fold metrics differ from the corresponding rows in
  `benchmark_results.csv`, confirming they are not the same run.

**These pilot files must not be interpreted as representative convergence trajectories
for the published benchmark.**

---

## Implications for Convergence Analysis

Full convergence analysis (objective score vs. HPO iteration, cumulative best score,
comparison of Random Search vs. Bayesian Optimization convergence speed) **cannot be
produced** from the completed benchmark run without rerunning.

### Rerunning is not recommended

The benchmark total runtime was substantial (the Adult/XGBoost/Bayesian/Fold-5
experiment alone took ~14.4 hours). A full rerun solely to recover iteration
histories is not warranted at this stage.

### Future runs will persist histories

`scripts/run_full_benchmark.py` has been updated (post-benchmark) to save each
experiment's full iteration history to:

```
results/history/<dataset>_<model>_<optimizer>_fold<N>.json
```

Each file will contain:

```json
{
  "dataset": "...",
  "outer_fold": N,
  "model": "...",
  "optimizer": "...",
  "best_params": { ... },
  "best_objective_score": ...,
  "history": [
    {
      "iteration": 1,
      "optimizer": "random|bayesian",
      "<hyperparam_1>": ...,
      "...",
      "mean_accuracy": ...,
      "mean_balanced_accuracy": ...,
      "mean_f1": ...,
      "mean_roc_auc": ...,
      "mean_demographic_parity_difference": ...,
      "mean_equal_opportunity_difference": ...,
      "objective_score": ...
    },
    ...
  ]
}
```

This fix is already committed. Any future benchmark rerun (e.g., with additional
HPO methods or datasets) will automatically produce 60 history files enabling
full convergence analysis.

---

## Recommended Reporting Language

When describing convergence in any manuscript or report:

> Per-iteration HPO convergence trajectories were generated internally during the
> benchmark but were not persisted to disk in the original run due to a logging
> omission in the benchmark driver script. Full convergence analysis across all
> 60 experimental conditions is therefore not available. Two pilot-run histories
> (Adult dataset, Random Forest, Fold 1, both optimizers, 10 iterations each)
> exist but originate from a preliminary search with a different hyperparameter
> space and should not be treated as representative of the published results.
> The benchmark driver has since been corrected to persist per-experiment histories
> for all future runs.

---

## Pilot History Contents (for Reference Only)

The two available pilot histories cover the following iteration counts and
objective score ranges (inner-CV balanced accuracy, fairness weight = 0):

### Adult / Random Forest / Random Search / Fold 1 (Pilot)

| Iteration | `n_estimators` | `max_depth` | Objective Score (BA) |
|---|---|---|---|
| 1 | 50 | 20 | 0.7547 |
| 2 | 50 | 20 | 0.7499 |
| 3 | 50 | 10 | 0.7273 |
| 4 | 50 | 20 | 0.7683 |
| 5 | 50 | None | **0.7763** (best) |
| 6 | 50 | 10 | 0.7474 |
| 7 | 200 | 20 | 0.7651 |
| 8 | 100 | 5 | 0.6661 |
| 9 | 100 | 20 | 0.7564 |
| 10 | 100 | 10 | 0.7259 |

Best found at iteration 5 of 10.

### Adult / Random Forest / Bayesian / Fold 1 (Pilot)

| Iteration | `n_estimators` | `max_depth` | Objective Score (BA) |
|---|---|---|---|
| 1 | 169 | 6 | 0.7096 |
| 2 | 117 | 5 | 0.6686 |
| 3 | 71 | 14 | 0.7579 |
| 4 | 191 | 3 | 0.5580 |
| 5 | 142 | 3 | 0.5579 |
| 6 | 110 | 4 | 0.6083 |
| 7 | 64 | 14 | 0.7550 |
| 8 | 120 | 18 | 0.7629 |
| 9 | 52 | 19 | **0.7661** (best) |
| 10 | 52 | 7 | 0.7209 |

Best found at iteration 9 of 10. Note the markedly higher variance across
iterations compared to random search, consistent with Bayesian optimization
exploring more of the space before exploiting good regions.

> **Caveat**: The pilot search space included `n_estimators` up to 200, whereas
> the final benchmark capped it at 100. These pilot results are therefore not
> directly comparable to the published benchmark results.
