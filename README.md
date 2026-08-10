# Titanic - Beginner ML Project (Python)

Project: Predict survival on the Titanic using a simple, reproducible pipeline.

What this repo contains
- notebooks/01_titanic_baseline.py — Jupyter-friendly script with EDA, preprocessing, baseline model, evaluation, and saved model.
- src/data.py — data loading and minimal cleaning utilities
- src/features.py — feature engineering helpers
- src/model.py — training, evaluation, and model saving functions
- requirements.txt — Python dependencies
- run_train.sh — simple shell script to run training
- .gitignore, LICENSE (MIT)

Dataset
- This project expects the Kaggle Titanic dataset (train.csv, test.csv).
- Download from: https://www.kaggle.com/c/titanic/data
- Place `train.csv` and `test.csv` in the `data/` folder.

Quick start (local)
1. Create a virtual environment:
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
2. Install dependencies:
   pip install -r requirements.txt
3. Ensure data/train.csv and data/test.csv exist.
4. Run the baseline training & evaluation:
   bash run_train.sh
   This trains a baseline model, prints metrics, and writes `models/baseline.pkl`.

What to explore next
- Improve feature engineering (title extraction from Name, age imputation by group)
- Hyperparameter tuning (GridSearch / RandomizedSearch)
- Add cross-validation and more robust metrics logging
- Convert notebook to a reproducible pipeline (MLflow / DVC)

Project checklist
- [x] Problem statement & success metric (accuracy / ROC‑AUC)
- [x] Data loading & simple cleaning
- [x] Baseline model (Logistic Regression / RandomForest)
- [x] Evaluation & basic error analysis
- [ ] Hyperparameter tuning
- [ ] Packaging (Docker + API) — optional



## Notes & Charts

- Purpose: Predict passenger survival on the Titanic using a clear, reproducible beginner pipeline (data → features → baseline model → evaluation).  
- Baseline model: RandomForest with minimal preprocessing (median age/fare imputation, one‑hot for Sex/Embarked).  
- Reproducibility: random_state = 42 used for train/test split and model where applicable. Models and encoder are saved to `models/`.  
- Data: place `train.csv` and `test.csv` in `data/` (do not commit data — `data/` is in .gitignore).  
- Next steps: add feature engineering (Title, FamilySize), stratified k‑fold CV, hyperparameter tuning, model explainability (SHAP), and a deployment endpoint (FastAPI + Docker).

### Charts & Plots (what to include in the notebook / README)

| Chart | Purpose | Where (notebook/script) | Plot type | Quick interpretation tips |
|---|---:|---|---|---|
| Survival rate by Pclass | Show class effect on survival | notebooks/01_titanic_baseline.py cell: "Survival by Pclass" | Bar plot | Lower classes typically have lower survival — check magnitude and sample sizes. |
| Survival rate by Sex | Show gender effect | same | Bar plot | Compare male vs female survival; large gap indicates strong predictor. |
| Age distribution (survived vs not) | Understand age influence | same | Kernel density / histogram | Look for age ranges with better survival; useful for imputation strategies. |
| Fare distribution by survival | See whether fare (proxy for socio‑economic) matters | same | Boxplot / violin | High fares may correlate with higher survival; inspect outliers. |
| Survival by Embarked | Check port effect | same | Bar plot | Compare survival rates across C/Q/S; used for feature importance. |
| Family size vs survival | Family effects (alone vs family) | same | Bar plot / line | Create FamilySize = SibSp + Parch + 1 and IsAlone flag; interpret social effects. |
| Title (extracted from Name) vs survival | Capture social/status signals | feature engineering cell | Bar plot | Titles like 'Mrs', 'Miss', 'Mr', 'Master' often have different survival rates. |
| Age vs Fare colored by survival | Multivariate view | same | Scatter (alpha) | Helps spot clusters (e.g., young/low fare) and their survival patterns. |
| Correlation heatmap (numeric features) | Quickly see feature correlations | EDA cell | Heatmap | Use to detect collinearity and choose features for models. |
| Feature importance (RandomForest) | Model explainability | training cell (after training) | Bar plot | Identify top predictors (Sex, Pclass, Age, Fare, Title). |
| Confusion matrix | Model performance diagnostics | evaluation cell | Heatmap / matrix | Inspect false positives/negatives to guide error analysis. |
| ROC curve & AUC | Discriminative performance | evaluation cell | ROC curve | Check AUC for ranking quality; compare models on the same plot. |

Tip: Add short captions under each plot in the notebook describing the takeaway (1–2 sentences). That makes the README and notebook more friendly for reviewers.


