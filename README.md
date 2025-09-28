# Demo Code Repository – Past Experience

This repository collects demo code snippets from previous professional experiences.  
It is **not a production system** – just a showcase of past projects and implementations.

---

## Projects

### 🔎 `search_ops_app`
- **Description**: Streamlit app surfacing groups of relevant searches based on conversion/click performance.
- **GitHub**: [search_ops_app](https://github.com/hafedrhouma-lab/code/tree/main/search_ops_app)

---

### 📦 `data-ml-pipeline`
- **Description**: Orchestrates model training and scheduled batch inference with experiment tracking in MLflow.
- **My ownership**: [`projects/vendor_ranking/two_tower_v1`](https://github.com/hafedrhouma-lab/code/tree/main/data-ml-pipelines/projects/vendor_ranking)
- **GitHub**: [data-ml-pipelines](https://github.com/hafedrhouma-lab/code/tree/main/data-ml-pipelines/projects/vendor_ranking)

---

### ☁️ `ACE`
- **Description**: Kubernetes (GKE) service for online ML model serving on GCP.
- **My ownership**: [`item_lifecycle`](https://github.com/hafedrhouma-lab/code/tree/main/Ace/item_lifecycle)
- **GitHub**: [ACE](https://github.com/hafedrhouma-lab/code/tree/main/Ace/item_lifecycle)

---

### 🏅 `kaggle_competition`
- **Goal**: Predict whether a session belongs to `user_id = 0`.
- **Imbalance**: ~800 positives out of ~160k sessions.
- **Why accuracy/ROC-AUC can mislead**: With so few positives, a model can “look good” while missing the target entirely.
- **Optimization focus**: maximize **recall** (catch user-0) while also maximizing **precision** (limit false alerts).

---

### 📈 `data-timeseries-forecast-tool`
- **Goal**: Forecast using a **mix** of calendar and reactive (year-to-year) components.
- **Calendar component**: captures seasonality, holidays, special events; uses **exponential smoothing** to produce a baseline level.
- **Reactive component**: year-to-year changes to remain adaptable.
- **Mixing strategy**: final forecast is a **weighted combination**; prediction is baseline × effect, balancing stability (calendar) and adaptability (reactive).

---

### 🏷️ `data-item-tagging-prompt`
- **Description**: Designed prompts to enrich Food delivery app's taxonomy by generating item-level tags with OpenAI models.  
- **Scope**: Tagging both **food** and **grocery** items to simplify search/discovery across thousands of SKUs.  
- **Tech**: Prompt engineering, async OpenAI API calls, BigQuery integration.  
- **GitHub**: [data-item-tagging-prompt](https://github.com/hafedrhouma-lab/code/tree/main/data-item-tagging-prompt)

---

### 🤖 `data-item-tagging-model`
- **Description**: Trained a **RoBERTa classifier** to predict item tags using datasets labeled via LLM outputs (or human annotation).  
- **Scope**: Automates item tagging at scale for the food and non-food categories.  
- **Tech**: PyTorch, scikit-learn, BigQuery, model training & prediction CLI.  
- **GitHub**: [data-item-tagging-model](https://github.com/hafedrhouma-lab/code/tree/main/data-item-tagging-model)

---

## Notes
- Code here is for **demonstration** only.
- Large artifacts and secrets are removed/ignored to keep the repo lightweight and safe.
