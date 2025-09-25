# Session Classification Pipeline (Time-Aware · Imbalanced · Thresholded)

Detect **user-0** sessions from raw web activity using:

- **Time-based train/test split** (no leakage from the future)
- **Forward-chaining cross-validation** (expanding window)
- **Recall-targeted thresholding** (optimize precision subject to recall ≥ target)
- **Model families:** Logistic Regression and LightGBM

---

## Project Structure
```
├── README.md                              # Project overview & setup
├── artifacts/                             # Outputs produced by CLIs (models, metrics, plots, preds)
│   ├── eval_grid/
│   │   └── evaluate_best_lgbm_20250917T230657Z.json   # Eval at multiple manual thresholds
│   ├── eval_stored/
│   │   └── evaluate_best_lgbm_20250917T230657Z.json   # Eval using stored/recomputed threshold
│   ├── fi/                                # Feature-importance CSVs/plots
│   │   ├── lgbm_shap_last.csv
│   │   └── lgbm_shap_last.png
│   ├── hpsearch_run_final2/               # One HP-search run’s artifacts
│   │   ├── best_lgbm_20250917T230657Z.joblib          # Persisted sklearn Pipeline + threshold
│   │   ├── lgbm/
│   │   │   └── cv_summary.json            # CV metrics & best params for LightGBM
│   │   ├── logreg/
│   │   │   └── cv_summary.json            # CV metrics & best params for Logistic Regression
│   │   └── summary_20250917T230657Z.json  # Final TEST metrics & chosen-threshold metadata
│   ├── predictions_lgbm_best_automatic_thr_last.csv   # Predictions w/ stored (auto) threshold
│   └── predictions_lgbm_best_thr0.21_last.csv         # Predictions w/ manual threshold = 0.21
├── cli/                                   # Thin CLI wrappers (interface only)
│   ├── evaluate.py                        # Evaluate saved artifact on TRAIN/TEST (auto/manual thr)
│   ├── feature_importance.py              # Dump/plot importances (built_in | permutation | shap)
│   ├── hp_search.py                       # HP search + forward-chaining CV + robust threshold
│   └── predict.py                         # Score new data → CSV with probs/labels + wide inputs
├── conda_environment.yml                  # Reproducible env (Py 3.11, sklearn, lightgbm, etc.)
├── data/
│   ├── __init__.py                        # Exports: PrepareConfig, prepare_dataset_raw, load_raw_df
│   ├── config.py                          # PrepareConfig dataclass
│   ├── datasets/
│   │   ├── dataset.json                   # Example training data (time-based split uses `date`)
│   │   └── verify.json                    # Example inference data for `cli/predict.py`
│   ├── io.py                              # Raw data I/O helpers (e.g., JSON→DataFrame)
│   ├── prepare.py                         # Feature engineering + prepare_dataset_raw (no leakage)
│   └── split.py                           # Time-aware split helpers (cutoffs, etc.)
├── exploration_analysis_find_Joe.pdf      # Exploratory analysis (optional)
├── model/
│   ├── __init__.py                        # Model package exports
│   ├── algorithms.py                      # Base estimators (make_logreg, make_lgbm)
│   ├── cv.py                              # ForwardTimeSplit (expanding-window CV)
│   ├── evaluate.py                        # Metrics, precision@recall scorer, robust thresholding
│   ├── infer/
│   │   ├── evaluate.py                    # Helpers used by CLI evaluate (auto/manual thresholds)
│   │   ├── importance.py                  # Importance strategies (built_in, permutation, shap)
│   │   ├── predict.py                     # Predict helpers (proba/labels, CSV shaping)
│   │   ├── prep.py                        # Inference-time shaping (expand_like_training, labels)
│   │   ├── registry.py                    # Importance strategy registry (decorators)
│   │   └── runner.py                      # Orchestrator: loads strategies, runs & saves outputs
│   ├── repository.py                      # ArtifactRepository: save/load Pipeline + threshold
│   ├── train.py                           # Shared training utilities (reused by CLIs)
│   └── training/
│       ├── hpsearch.py                    # HP search orchestration (pure, unit-testable functions)
│       ├── pipeline.py                    # Pipelines & search spaces (registered model families)
│       └── registry.py                    # Model-family registry (decorators)
├── notebooks/
│   └── exploratory.ipynb                  # Optional EDA notebook
├── pyproject.toml                         # Tooling config (ruff, mypy, pytest)
├── pytest.ini                             # Pytest options
├── tests/                                 # Unit/integration tests (fast & deterministic)
│   ├── conftest.py                        # Fixtures (synthetic data, small PrepareConfig)
│   ├── test_evaluate_math.py              # Threshold math & CV thresholding
│   ├── test_infer_prep.py                 # Inference prep & label extraction
│   ├── test_prepare.py                    # Data shaping & FE sklearn-compat (clone/fit/transform)
│   ├── test_registries.py                 # Registries populated (training & infer)
│   └── test_repository.py                 # Artifact roundtrip (save/load Pipeline + threshold)
└── utils/
    ├── __init__.py                        # Utils exports
    └── logger.py                          # Structured logging (namespaced loggers)
```
---

## Environment
A reproducible conda spec is provided in `conda_environment.yml` (Python 3.11, NumPy, pandas, scikit-learn, LightGBM, etc.).

```bash
conda env create -f conda_environment.yml
conda activate toptal_test
# If needed (depending on your channels):
# pip install lightgbm
```

---

## Data Format
Each JSON row is one session:

```json
{
  "browser": "Firefox",
  "os": "Ubuntu",
  "locale": "ru-RU",
  "gender": "m",
  "location": "Singapore/Singapore",
  "sites": [
    {"site": "mail.google.com", "length": 63},
    {"site": "vk.com", "length": 126}
  ],
  "time": "13:06:00",
  "date": "2019-04-22",
  "user_id": 0
}
```

During feature engineering we:
- expand `sites` → `site_1..site_15`, `length_1..length_15`
- split `location` → `(country, city)`
- build **TF-IDF** of sites, **signature-site flags** for user-0, **time features** (hour/DOW/month, cyclic hour),
- compute **length stats** (sum/mean/median/std/max/min, `n_sites`, `n_visits`), and **one-hot** meta (`browser`/`os`/`locale`/`gender`/`country`/`city`).

---

## How Hyper-Parameter Search + CV Works

We preserve causality by splitting on date:

- **TRAIN:** rows with `date < cutoff`
- **TEST:** rows with `date >= cutoff` (held-out, never used for HP search)

On TRAIN, we run `RandomizedSearchCV` with `ForwardTimeSplit` (expanding window). With `n_splits = 3`, the folds look like this (time flows →):

```
T1  T2  T3  T4  T5  T6  T7  T8  T9
|---train---| val
|------train------|      val
|----------train----------| val
```

- **Fold 1:** Train on `[T1..T6]`, validate on `[T7]`
- **Fold 2:** Train on `[T1..T7]`, validate on `[T8]`
- **Fold 3:** Train on `[T1..T8]`, validate on `[T9]`

For each model family (LogReg, LGBM) and each sampled hyper-parameter set, we:

1. Fit on the train part of each fold.
2. Score on the val part with two metrics:
   - **Average Precision (ap)**
   - **Precision@Recall≥target (prec_at_rec)** — our refit metric → prioritizes precision given a minimum recall (e.g., 100%).

We then aggregate per-fold scores and pick the hyper-params that maximize `prec_at_rec`. Finally, we refit the winning model on full TRAIN and choose a decision threshold:

- For each fold, find thresholds that achieve `recall ≥ target`; pick the one with **maximum precision**.
- Aggregate per-fold thresholds with **median** (or **p25/p10** for more recall-safety).

### High-level flow (two models)
```
           ┌─────────────┐            ┌──────────────┐
Raw JSON → │  FE (fit)   │ → X_train  │ ForwardTimeCV │
           └─────┬───────┘            └──────┬───────┘
                 │                           │
         ┌───────▼─────────┐         ┌───────▼─────────┐
         │ LogReg params   │         │  LGBM params    │   (random search)
         └───────┬─────────┘         └───────┬─────────┘
                 │(fit+score folds)          │(fit+score folds)
                 └─────────┬─────────────────┘
                           ▼
                 pick best family by prec@rec
                           │
                     refit on TRAIN
                           │
                 robust threshold (median/p25/p10)
                           │
                        evaluate on TEST
                           ▼
                    save model+FE+threshold
```

---

## Commands

### 1) Hyper-parameter Search + Robust Threshold
Runs time-aware HP search for both models and saves the best model, the fitted feature engineer, and the chosen threshold.

```bash
python -m cli.hp_search \
  --raw data/datasets/dataset.json \
  --artifacts artifacts/hpsearch_run_final2 \
  --cutoff 2019-01-01 \
  --n_splits 3 --n_iter 4 \
  --recall_target 0.52 \
  --thr_agg median \
  --thr_use_max_precision \
  --n_jobs 8
```

**Key args**
- `--cutoff` — TRAIN < cutoff, TEST >= cutoff
- `--n_splits` — number of forward-chaining folds
- `--n_iter` — random hyper-param samples per model family
- `--recall_target` — minimum recall to enforce when optimizing precision
- `--thr_agg` — aggregate per-fold thresholds: `median` (default), `p25`, `p10`
- `--thr_use_max_precision` — in each fold, pick threshold with max precision among those meeting the recall target

**Outputs:**
- `artifacts/hpsearch_run_final2/summary_*.json` — TEST metrics & chosen threshold
- `artifacts/hpsearch_run_final2/best_<model>_*.joblib` — best model (+ FE + threshold)
- `artifacts/hpsearch_run_final2/<model>/cv_summary.json` — per-model CV performance & best params

### 2) Evaluate a Saved Model on TRAIN/TEST
Use the stored threshold (default), manual thresholds, or recompute a robust threshold (same way as HP search).

**Stored threshold:**
```bash
python -m cli.evaluate \
  --model artifacts/hpsearch_run_final2/best_lgbm_20250917T230657Z.joblib \
  --raw data/datasets/dataset.json \
  --cutoff 2019-01-01 \
  --out_dir artifacts/eval_stored \
```

**Manual threshold grid:**
```bash
python -m cli.evaluate \
    --model artifacts/hpsearch_run_final2/best_lgbm_20250917T230657Z.joblib \
    --raw data/datasets/dataset.json \
    --cutoff 2019-01-01 \
    --out_dir artifacts/eval_grid \
    --thresholds 0.06,0.065,0.13,0.14,0.15
```

**Recompute robust threshold via CV:**
```bash
python -m cli.evaluate \
  --model artifacts/hpsearch_run_final2/best_lgbm_20250917T230657Z.joblib   \
  --raw data/datasets/dataset.json   \
  --cutoff 2019-01-01   \
  --out_dir artifacts/eval_grid   \
  --recompute_threshold   \
  --n_splits 5   \
  --recall_target 1.0   \
  --thr_agg median   \
  --thr_use_max_precision
```

### 3) Predict on New Data
Produces a CSV that includes all input metadata (wide format) plus **prob** and **label**.

**Using stored threshold:**
```bash
python -m cli.predict   \
  --model artifacts/hpsearch_run_final2/best_lgbm_20250917T230657Z.joblib   \
  --raw data/datasets/verify.json   \
  --out artifacts/predictions_lgbm_best_automatic_thr.csv
```

**Override threshold:**
```bash
python -m cli.predict   \
  --model artifacts/hpsearch_run_final2/best_lgbm_20250917T230657Z.joblib   \
  --raw data/datasets/verify.json   \
  --out artifacts/predictions_lgbm_best_thr0.26.csv   \
  --threshold 0.26
```

### 4) Feature Importance

Built-in importances (fast; LGBM split counts or |LogReg coef|) and optional plot using SHAP:
```bash
python -m cli.feature_importance \
  --model artifacts/hpsearch_run_final2/best_lgbm_20250917T230657Z.joblib \
  --method shap \
  --raw data/datasets/dataset.json \
  --cutoff 2019-01-01 \
  --out_csv artifacts/fi/lgbm_shap.csv \
  --topk 50 \
  --plot --plot_path artifacts/fi/lgbm_shap.png
```
---

## Notes on Imbalance & Thresholds
- Models use `class_weight="balanced"` (LogReg) or `scale_pos_weight` (LGBM) during training.
- We **don’t** assume `threshold = 0.5`.
- We select a threshold to meet `recall ≥ target`, and (optionally) maximize **precision** at that recall within each fold, then **aggregate thresholds**.
