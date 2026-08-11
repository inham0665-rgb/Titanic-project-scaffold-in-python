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

Where to paste this
- Insert this section into `README.md` (recommended after Quick start or as a new "Notes & Charts" section).  
- Optionally link to `NOTES.md` for more details: `Notes: [NOTES.md](NOTES.md)`.

---

## Execution examples — classification, regression, and advanced techniques

This section contains ready-to-run code snippets and explanations you can copy into scripts or notebooks. They show how to: train the baseline classifier, run cross-validation and hyperparameter tuning, run a regression example (predict `Fare`), compute SHAP explanations, and run a minimal FastAPI inference endpoint.

Prerequisites
- Python environment with packages from `requirements.txt` installed. Add these for extras:

```bash
pip install xgboost shap fastapi uvicorn[standard]
```

A. Classification — baseline (runable)

Command-line (existing train script):

```bash
# trains RandomForest baseline and saves models/baseline.pkl
python src/train.py --train-csv data/train.csv --output-dir models
```

Equivalent quick script (copy to scripts/train_baseline.py):

```python
# scripts/train_baseline.py
import joblib
from src.data import load_data, basic_cleaning
from src.features import extract_features
from src.model import train_and_evaluate

if __name__ == '__main__':
    df = load_data('data/train.csv')
    df = basic_cleaning(df)
    X, y, encoder = extract_features(df)
    model, acc, roc = train_and_evaluate(X, y, output_dir='models')
    # encoder saved by train script if present
```

Explanation
- train_and_evaluate performs a stratified holdout split (80/20) and trains a RandomForestClassifier. It prints accuracy, ROC-AUC, classification report, confusion matrix, and saves `models/baseline.pkl`.

B. Cross-validation & hyperparameter tuning (classification)

Example using RandomizedSearchCV (faster than grid search):

```python
# scripts/hyperparam_tune.py
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from src.data import load_data, basic_cleaning
from src.features import extract_features

df = load_data('data/train.csv')
df = basic_cleaning(df)
X, y, _ = extract_features(df)

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=12, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
search.fit(X, y)
print('Best params:', search.best_params_)
print('Best score:', search.best_score_)
```

Explanation
- Use stratified k-fold to avoid class imbalance issues. Score by ROC-AUC for ranking models when class imbalance exists. Save `search.best_estimator_` for deployment.

C. Regression example — predict Fare (toy problem)

Sometimes you want to practice regression on the same dataset (predict `Fare` given passenger features). This is a simple example using RandomForestRegressor.

```python
# scripts/regress_fare.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data import load_data, basic_cleaning
from src.features import extract_features

# load & prepare
df = load_data('data/train.csv')
df = basic_cleaning(df)
# target = Fare; drop rows where Fare missing
reg_df = df.dropna(subset=['Fare']).copy()
y = reg_df['Fare']
X, _y, encoder = extract_features(reg_df)

# If extract_features drops Fare, reconstruct feature matrix to include predictors only (ensure Fare not in X)
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)
print(f'Fare regression RMSE: {rmse:.3f}, R2: {r2:.3f}')
# Save model
joblib.dump(model, 'models/fare_regressor.pkl')
```

Explanation
- This is a toy regression task. It helps practice preprocessing, dealing with skewed targets (consider log-transform of Fare), and regression metrics (RMSE, R2).

D. Model explainability with SHAP (classification)

Quick snippet to compute SHAP values for the RandomForest baseline:

```python
# scripts/explain_shap.py
import joblib
import shap
import pandas as pd
from src.data import load_data, basic_cleaning
from src.features import extract_features

model = joblib.load('models/baseline.pkl')
df = load_data('data/train.csv')
df = basic_cleaning(df)
X, y, encoder = extract_features(df)
# Use a sample to speed up SHAP
X_sample = X.sample(200, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
# Summary plot (class 1)
shap.summary_plot(shap_values[1], X_sample)
```

Explanation
- SHAP provides local and global explanations. For tree models use TreeExplainer; for NN use DeepExplainer or KernelExplainer as appropriate.

E. Minimal FastAPI inference example

Create `app.py` with the following content:

```python
# app.py
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/baseline.pkl')
encoder = None
try:
    encoder = joblib.load('models/encoder.pkl')
except Exception:
    pass

@app.post('/predict')
def predict(payload: dict):
    # payload: dict of feature values matching train features
    df = pd.DataFrame([payload])
    # minimal preprocessing: align columns and encode as in training
    # if encoder is present, apply encoder to categorical columns
    # This example assumes the payload matches the model's expected columns
    preds = model.predict(df)
    probs = model.predict_proba(df)[:,1] if hasattr(model, 'predict_proba') else None
    return {'prediction': int(preds[0]), 'probability': float(probs[0]) if probs is not None else None}

# Run with:
# uvicorn app:app --reload --port 8000
```

Explanation
- For production, add input validation (pydantic models), robust preprocessing identical to training pipeline, and authentication.

F. Running the notebook

- The notebook-like script is at `notebooks/01_titanic_baseline.py`. You can:
  - Open it in VS Code and run cells with the Python extension, or
  - Convert to `.ipynb` with jupytext, or
  - Run it as a script: `python notebooks/01_titanic_baseline.py` (plots will open in supported environments).

G. Reproducibility & experiments

- Set `random_state=42` where applicable. Save models and encoders with `joblib.dump()` in `models/`.
- For experiment tracking use MLflow (quick example):

```bash
pip install mlflow
mlflow run . -P alpha=0.5
```

- For data versioning use DVC if you want to track large datasets outside Git.

---

If you want, I will:
- Commit this README addition to `main` now (commit message: "Add execution examples & code snippets to README").
- Or instead create separate script files under `scripts/` for each example and push them to the repo. The latter makes the examples immediately runnable; tell me which option you prefer.
