# Titanic - Beginner ML Project (Python)

Project: Predict survival on the Titanic using a simple, reproducible pipeline.

What this repo contains
- notebooks/01_titanic_baseline.py — Jupyter-friendly script with EDA, preprocessing, baseline model, evaluation, and saved model.
- src/data.py — data loading and minimal cleaning utilities
- src/features.py — feature engineering helpers
- src/model.py — training, evaluation, and model saving functions
- requirements.txt — Python dependencies
- run_train.sh — simple shell script to run training (classification)
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
4. Run the baseline training & evaluation (classification):
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

If you want, I can push this scaffold to a new GitHub repo for you or create a single notebook .ipynb file instead of the .py notebook.

---

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

## Classification — how to run, how it works, and interpreting feedback

### How to run (classification: predict `Survived`)

From the repo root (after you placed `data/train.csv`):

```bash
# Run the existing classification training script
python src/train.py --train-csv data/train.csv --output-dir models
```

This script runs a baseline RandomForestClassifier and saves the model to `models/baseline.pkl` (and `models/encoder.pkl` if an encoder is created).

### What the classification code does (pipeline)
1. Data loading: `src/data.py::load_data` reads CSV into a pandas DataFrame.
2. Basic cleaning: `src/data.py::basic_cleaning` fills missing Age with median and Embarked with mode, drops high-missing/noisy columns (Cabin, Ticket).
3. Feature extraction: `src/features.py::extract_features` selects relevant columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked), fills missing Fare, and one-hot encodes Sex/Embarked.
4. Train/test split: `src/model.py::train_and_evaluate` splits data (stratified) and trains RandomForestClassifier.
5. Evaluation: prints validation accuracy, ROC AUC (if probabilities available), classification report, and confusion matrix.
6. Artifacts: model and encoder are saved in `models/`.

### What feedback to look for (classification metrics)
- Accuracy: overall fraction of correct predictions. Useful as a baseline but can be misleading if classes are imbalanced.
- ROC AUC: measures ranking quality of predicted probabilities; values closer to 1.0 are better. Good when you care about ranking or imbalanced classes.
- Precision / Recall / F1: per-class measures. For `Survived=1` (positive class):
  - Precision: fraction of predicted survivors that actually survived (reducing false positives).
  - Recall: fraction of true survivors correctly predicted (reducing false negatives).
  - F1: harmonic mean of precision & recall; useful when seeking balance.
- Confusion matrix: shows true vs predicted counts to diagnose common error types.

Interpretation tips
- If recall for Survived is low, the model misses survivors (false negatives); consider adjusting class weights, threshold, or using a model that better captures minority class.
- If precision is low, many predicted survivors did not survive (false positives); consider precision-oriented tuning.
- Use feature importance or SHAP values to understand which features drive predictions (Sex, Pclass often dominate in Titanic).

---

## Regression — how to run, how it works, and interpreting feedback

This repository's main task is classification (Survived). We include a simple regression example that predicts `Fare` using the other passenger features. This is useful as an exercise in regression modeling and evaluation.

### How to run (regression: predict `Fare`)

From the repo root (after you placed `data/train.csv`):

```bash
python src/train_regression.py --train-csv data/train.csv --output-dir models
```

This will train a RandomForestRegressor to predict Fare and save the model to `models/regression_baseline.pkl`.

### What the regression code does (pipeline)
1. Data loading & cleaning: reuses `src/data.py` cleaning to fill Age/Embarked and drop noisy columns.
2. Target selection: sets `y = Fare` and removes Fare from the feature set before training.
3. Feature building: encodes categorical fields (Sex, Embarked) and keeps numeric fields (Pclass, Age, SibSp, Parch).
4. Train/test split: simple random split with a fixed random_state for reproducibility.
5. Model: RandomForestRegressor is trained on the training set.
6. Evaluation: prints RMSE and MAE on the validation set and saves the model artifact.

### What feedback to look for (regression metrics)
- RMSE (Root Mean Squared Error): penalizes large errors more heavily; same units as target (fare). Lower is better.
- MAE (Mean Absolute Error): average absolute error; more robust to outliers than RMSE.

Interpretation tips
- Compare RMSE/MAE to the typical Fare scale in your data (e.g., median Fare). A small RMSE relative to median Fare indicates reasonable predictions.
- Examine residuals (prediction - true) to find patterns or heteroscedasticity (errors that grow with Fare). Consider log-transforming Fare if distribution is skewed.
- Feature importance: identify which features contribute most to Fare prediction; high importance for Pclass / Title may indicate fare is strongly tied to class and social status.

---

If you'd like, I can also:
- Add a CLI flag to `src/train.py` to switch between classification and regression modes instead of a separate script.
- Add plotting code in the notebook to display residuals, predicted vs actual, and the regression feature importances.
- Add unit tests and a small GitHub Actions workflow to run a smoke test on commit.
