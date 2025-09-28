# Demo Code Repository – Past Experience

This repository collects demo code snippets from previous professional experiences.  
It is **not a production system** – just a showcase of past projects and implementations.

---

## Projects

### 🔎 `search_ops_app`
- **Description**: Streamlit app to monitor and improve grocery search performance by surfacing *best/worst* query groups with CTR/CVR/Clicks and drill-downs by country/area.
- **Scope**:
  - Visualize search KPIs (CTR, CVR, results, clicks) at country/area granularity
  - Group queries into performance clusters (best/worst) for homepage and in-vendor search
  - Enable analysts to filter, segment, and export insights
- **Tech**:
  - **Streamlit** (multi-page app, custom CSS), **Pandas**
  - **BigQuery** (+ templated SQL via **Jinja2** `.sql.j2`)
  - Query clustering (quantile-based), session-level aggregations
  - **Docker** (local), **GCP Cloud Build**/**Cloud Run** (deployment-ready)
  - Config files, logging, unit tests
- **GitHub**: [search_ops_app](https://github.com/hafedrhouma-lab/code/tree/main/search_ops_app)

---

### 📦 `data-ml-pipeline`
- **Description**: Mono-repo scaffold for ML projects with **MLflow** tracking/registry, experiment logging, and paths to both **batch** (Airflow) and **online** serving (ACE).
- **Scope**:
  - Project scaffolding (`get_started.py`) → standard structure for new models
  - Train/evaluate/track models; register versions to MLflow with aliases
  - Batch inference orchestration and scheduling (Airflow)
  - Online serving hand-off through ACE model wrapper
- **Tech**:
  - **MLflow** (runs, artifacts, model registry/aliases)
  - Reusable **base** layer (DB utils/**BigQuery**, FS, perf metrics)
  - **Airflow** jobs via `schedule.yaml`
  - Packaging & environments (conda/requirements), **pytest**
  - CI/CD-friendly layout (e.g., GitHub Actions / **Cloud Build** / Travis / Drone / CircleCI)
- **My ownership**: [`projects/vendor_ranking/two_tower_v1`](https://github.com/hafedrhouma-lab/code/tree/main/data-ml-pipelines/projects/vendor_ranking)
- **GitHub**: [data-ml-pipelines](https://github.com/hafedrhouma-lab/code/tree/main/data-ml-pipelines/projects/vendor_ranking)

---

### ☁️ `ACE`
- **Description**: “Data Serving (codename Ace)” – services for **online ML model serving** on **Kubernetes (GKE)** with reproducible local envs and CI support.
- **Scope**:
  - API services for real-time inference; local **Postgres** + client app for integration tests
  - Env management with **direnv/pyenv/venv**, dependency locking with **pip-tools**
  - Model packaging & release flow (e.g., **BentoML** builds)
  - Profiling & performance tuning (**py-spy**, k6)
- **Tech**:
  - **GKE / Kubernetes**, **Docker**, docker-compose
  - **BentoML** packaging, structured logging, **ruff** linting
  - CI pipelines (Cloud Build / Travis / Drone / CircleCI ready)
  - Make/Just helpers, Lima (containerd alternative)
- **My ownership**: [`item_lifecycle`](https://github.com/hafedrhouma-lab/code/tree/main/Ace/item_lifecycle)
- **GitHub**: [ACE](https://github.com/hafedrhouma-lab/code/tree/main/Ace/item_lifecycle)

---

### 🏅 `kaggle_competition`
- **Goal**: Classify whether a session belongs to **user_id = 0** under strong class imbalance (~800/160k).
- **Scope**:
  - Time-aware evaluation: **time-based split** + **forward-chaining CV**
  - Robust **threshold selection**: maximize precision subject to recall ≥ target; aggregate per-fold thresholds
  - Artifacts saved (models, metrics, SHAP plots, predictions) + fully scriptable CLIs
- **Tech**:
  - **LightGBM**, **Logistic Regression** (sklearn pipelines)
  - Feature engineering: **TF-IDF of sites**, signature-site flags, cyclic time features, session stats
  - **SHAP** importance, permutation importance
  - **conda** env, **pytest**, structured logging
- **Why accuracy/ROC-AUC can mislead**: severe imbalance → focus on recall/precision & PR-AUC.

---

### 📈 `data-timeseries-forecast-tool`
- **Goal**: Forecast via a **hybrid** of calendar effects and reactive YoY dynamics.
- **Scope**:
  - **Calendar** component: seasonality/holidays/special events modeled as multiplicative **effects**
  - **Reactive** component: YoY growth (“last_year_value × increase_rate”)
  - Final forecast: **baseline (level line via exponential smoothing) × effect, weighted with reactive forecast**
- **Tech**:
  - **Exponential Smoothing** (level smoothing; `span`, `beta`, `damped`)
  - Effect modeling for **date-of-year**, **holidays**, **special events**
  - **Outlier handling** & anomaly masking (COVID/bugs/events YAML)
  - **Pandas** package API; plots for effects & forecasts
- **Package usage**: `pip install .` → `CalendarForecast` / `ReactiveForecast` classes.

---

### 🏷️ `data-item-tagging-prompt`
- **Description**: Designed prompts to enrich a food delivery app taxonomy by generating item-level tags with OpenAI.
- **Scope**:
  - Tag **food/non-food** items to improve search/discovery
  - Async prompt execution; dataset simplification rules; BigQuery integration
- **Tech**:
  - Prompt engineering; OpenAI async calls
  - **BigQuery**, dataset preparation, local JSON caching
- **GitHub**: [data-item-tagging-prompt](https://github.com/hafedrhouma-lab/code/tree/main/data-item-tagging-prompt)

---

### 🤖 `data-item-tagging-model`
- **Description**: **RoBERTa-base** fine-tuned for binary/multilabel classification (vegetarian, gluten_free, sugar_free etc) from title + description.
- **Scope**:
  - Clean/concat text → tokenize → fine-tune → evaluate (accuracy/precision/recall/**F1**)
  - Deployable CLI for training/predict
- **Tech**:
  - **Transformers (Hugging Face)** + **PyTorch**
  - **Multi-head self-attention** (12 layers, 12 heads, hidden 768)
  - BigQuery ingestion, scikit-learn, experiment logging
- **GitHub**: [data-item-tagging-model](https://github.com/hafedrhouma-lab/code/tree/main/data-item-tagging-model)

---

## Notes
- Code here is for **demonstration** only.
- Large artifacts and secrets are removed/ignored to keep the repo lightweight and safe.
