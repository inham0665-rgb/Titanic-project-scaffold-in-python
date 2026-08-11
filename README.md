
# Titanic ML Project Scaffold :

Project scaffold to learn end-to-end supervised ML using the Kaggle Titanic
dataset. The goal is a clean, reproducible pipeline you can run on a laptop:
data → EDA → features → baseline models (classification & regression) →
evaluation → simple artifacts.

This repo is beginner-friendly and structured so you can iterate quickly and
demonstrate results in a portfolio.
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


## Contents / What's included
- `notebooks/01_titanic_baseline.py` — notebook-style script for EDA, plots, and interactive work
- `src/`
  - `data.py` — data loading & basic cleaning helpers
  - `features.py` — basic feature extraction and encoding (used by classification)
  - `model.py` — classification training & evaluation utilities
  - `train.py` — CLI entrypoint for classification (predict `Survived`)
  - `train_regression.py` — CLI entrypoint for regression (predict `Fare`)
- `requirements.txt` — Python dependencies
- `run_train.sh` — convenience script to run classification training
- `NOTES.md` — project notes, tips, and links
- `LICENSE` — MIT license
- `.gitignore` — ignores venv, datasets, models, notebook checkpoints

PIPELINE OVERVIEW:
<img width="694" height="238" alt="Screenshot 2026-08-11 011729" src="https://github.com/user-attachments/assets/b4e6bd4d-b8ff-4f08-a680-30362f0e7885" />

## Quick start (local)

```bash
# Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download data
Download `train.csv` and `test.csv` from Kaggle:
https://www.kaggle.com/c/titanic/data
Place them into the `data/` folder (create it if it doesn't exist).

### Run baseline classification (predict `Survived`)
```bash
bash run_train.sh
# or
python src/train.py --train-csv data/train.csv --output-dir models
```

### Run baseline regression (predict `Fare`)
```bash
python src/train_regression.py --train-csv data/train.csv --output-dir models
```

After training, models are saved into `models/`:
- Classification: `models/baseline.pkl`
- Regression: `models/regression_baseline.pkl`
- <img width="689" height="148" alt="Screenshot 2026-08-11 011930" src="https://github.com/user-attachments/assets/541add9b-3c9d-4e08-8bd7-b5e0cd6ed9f7" />


### Run the EDA notebook / chart generator
```bash
python notebooks/01_titanic_baseline.py
```
This generates all charts listed below into `outputs/` as PNGs.

## Project overview (step-by-step)

1. **Data loading** — `src/data.py::load_data` reads CSV into a DataFrame;
   `basic_cleaning` fills `Age` with the median, `Embarked` with the mode, and
   drops `Cabin`/`Ticket`.
2. **EDA** — `notebooks/01_titanic_baseline.py` plots survival by `Pclass`,
   `Sex`, `Embarked`; `Age`/`Fare` distributions; family-size effects;
   correlation heatmap; missing-value summary.
3. **Feature engineering** — `src/features.py` keeps
   `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`, fills missing `Fare` with
   the median, and one-hot encodes `Sex`/`Embarked`. Suggested improvements:
   extract `Title` from `Name` and impute `Age` by title group, add
   `FamilySize`/`IsAlone`, bin or log-transform skewed features.
4. **Modeling & evaluation**
   - Classification: `src/train.py` trains a `RandomForestClassifier` on
     `Survived` with a stratified train/validation split; prints accuracy,
     ROC AUC, classification report, confusion matrix.
   - Regression: `src/train_regression.py` trains a `RandomForestRegressor`
     to predict `Fare`; prints RMSE, MAE.
   - Artifacts (models, encoders) are saved to `models/`.

## Interpreting results
**Classification metrics**
- Accuracy: overall correctness, can be misleading with imbalance.
- Precision/Recall/F1: inspect for `Survived=1`; tune threshold or
  `class_weight` if needed.
- ROC AUC: ranking quality of predicted probabilities.
- Confusion matrix: analyze FP/FN to guide fixes
<img width="681" height="193" alt="Screenshot 2026-08-11 012121" src="https://github.com/user-attachments/assets/ffb13e36-5ff1-4260-afeb-c2a91c6c136f" />


**Regression metrics**
- RMSE: penalizes large errors; same units as `Fare`.
- MAE: average absolute error, robust to outliers.
- Compare errors to median `Fare` to judge practical significance; plot
  residuals and predicted vs actual.
  
 : DATA CLEANING:
 <img width="693" height="231" alt="Screenshot 2026-08-11 012308" src="https://github.com/user-attachments/assets/21ae28db-9633-49ec-a844-394ab867aa1d" />


## Charts included
1. Survival rate by `Pclass` (bar)
2. Survival rate by `Sex` (bar)
3. Age distribution by survival (KDE)
4. Fare distribution by survival (box)
5. Survival by `Embarked` (bar)
6. `FamilySize` vs survival (line)
7. `Title` (from `Name`) vs survival (bar)
8. Age vs Fare colored by survival (scatter)
9. Correlation heatmap (numeric features)
10. Feature importance (RandomForest)
11. Confusion matrix + ROC curve

Sample outputs from a demo run are in `outputs/`.

## Experiments & reproducibility
- Random state fixed (42) in train/test splits and model constructors.
- Artifacts saved in `models/`.
- For experiment tracking, consider MLflow or DVC (see `NOTES.md`).
- `data/` is kept out of version control (`.gitignore`); download
  instructions are documented above instead.

## Next improvements (good issues for contributors)
- Implement `Title` extraction and `Age` imputation by `Title`.
- Add `FamilySize` and `IsAlone` features to the baseline classifier.
- Replace single holdout with stratified k-fold CV and log fold metrics.
- Try alternative models: `LogisticRegression`, `XGBoost`, `LightGBM`.
- Add a SHAP explainability notebook and plots.
- Add a FastAPI inference service + Dockerfile for deployment.
- Add a GitHub Actions workflow to run linting and a smoke training job.

## Tests & CI (suggested)
- Add small unit tests for data loaders and feature functions (`pytest`).
- Add a GitHub Actions workflow: lint (`ruff`/`flake8`) + a smoke training
  job (train on the first N rows, assert metrics exist and the model file is
  created).

## Contributing
Fork the repo, create a feature branch, open a PR with a clear description
and tests where applicable. Use the Issues page to propose changes or
request features. Keep changes small and focused; update `NOTES.md` and this
README when adding new scripts or functionality.

FUTURE ENGENEERING:
<img width="988" height="168" alt="image" src="https://github.com/user-attachments/assets/eb375474-f1f5-46ae-a7be-5459ab99e45c" />

## License
MIT — see `LICENSE`.

## Classification — how to run, how it works, and interpreting feedback

### How to run (classification: predict `Survived`)

From the repo root (after you placed `data/train.csv`):

```bash
# Run the existing classification training script
python src/train.py --train-csv data/train.csv --output-dir models
This script runs a baseline RandomForestClassifier and saves the model to models/baseline.pkl (and models/encoder.pkl if an encoder is created).

What the classification code does (pipeline)
Data loading: src/data.py::load_data reads CSV into a pandas DataFrame.
Basic cleaning: src/data.py::basic_cleaning fills missing Age with median and Embarked with mode, drops high-missing/noisy columns (Cabin, Ticket).
Feature extraction: src/features.py::extract_features selects relevant columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked), fills missing Fare, and one-hot encodes Sex/Embarked.
Train/test split: src/model.py::train_and_evaluate splits data (stratified) and trains RandomForestClassifier.
Evaluation: prints validation accuracy, ROC AUC (if probabilities available), classification report, and confusion matrix.
Artifacts: model and encoder are saved in models/.
What feedback to look for (classification metrics)
Accuracy: overall fraction of correct predictions. Useful as a baseline but can be misleading if classes are imbalanced.
ROC AUC: measures ranking quality of predicted probabilities; values closer to 1.0 are better. Good when you care about ranking or imbalanced classes.
Precision / Recall / F1: per-class measures. For Survived=1 (positive class):
Precision: fraction of predicted survivors that actually survived (reducing false positives).
Recall: fraction of true survivors correctly predicted (reducing false negatives).
F1: harmonic mean of precision & recall; useful when seeking balance.
Confusion matrix: shows true vs predicted counts to diagnose common error types.
Interpretation tips

If recall for Survived is low, the model misses survivors (false negatives); consider adjusting class weights, threshold, or using a model that better captures minority class.
If precision is low, many predicted survivors did not survive (false positives); consider precision-oriented tuning.
Use feature importance or SHAP values to understand which features drive predictions (Sex, Pclass often dominate in Titanic).
Regression — how to run, how it works, and interpreting feedback
This repository's main task is classification (Survived). We include a simple regression example that predicts Fare using the other passenger features. This is useful as an exercise in regression modeling and evaluation.

How to run (regression: predict Fare)
From the repo root (after you placed data/train.csv):

bash
python src/train_regression.py --train-csv data/train.csv --output-dir models
This will train a RandomForestRegressor to predict Fare and save the model to models/regression_baseline.pkl.

What the regression code does (pipeline)
Data loading & cleaning: reuses src/data.py cleaning to fill Age/Embarked and drop noisy columns.
Target selection: sets y = Fare and removes Fare from the feature set before training.
Feature building: encodes categorical fields (Sex, Embarked) and keeps numeric fields (Pclass, Age, SibSp, Parch).
Train/test split: simple random split with a fixed random_state for reproducibility.
Model: RandomForestRegressor is trained on the training set.
Evaluation: prints RMSE and MAE on the validation set and saves the model artifact.
What feedback to look for (regression metrics)
RMSE (Root Mean Squared Error): penalizes large errors more heavily; same units as target (fare). Lower is better.
MAE (Mean Absolute Error): average absolute error; more robust to outliers than RMSE.
Interpretation tips

Compare RMSE/MAE to the typical Fare scale in your data (e.g., median Fare). A small RMSE relative to median Fare indicates reasonable predictions.
Examine residuals (prediction - true) to find patterns or heteroscedasticity (errors that grow with Fare). Consider log-transforming Fare if distribution is skewed.
Feature importance: identify which features contribute most to Fare prediction; high importance for Pclass / Title may indicate fare is strongly tied to class and social status.
If you'd like, I can also:

Add a CLI flag to src/train.py to switch between classification and regression modes instead of a separate script.
Add plotting code in the notebook to display residuals, predicted vs actual, and the regression feature importances.
Add unit tests and a small GitHub Actions workflow to run a smoke test on commit.
Code
src/train_regression.py

py
"""
Train a regression model to predict Fare using passenger features.
Usage:
    python src/train_regression.py --train-csv data/train.csv --output-dir models
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from src.data import load_data, basic_cleaning


def build_features_for_regression(df: pd.DataFrame):
    df = df.copy()
    # target
    y = df['Fare'] if 'Fare' in df.columns else None
    # feature columns
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    cols = [c for c in cols if c in df.columns]
    X = df[cols].copy()
    # fill numeric missing
    if 'Age' in X.columns:
        X['Age'] = X['Age'].fillna(X['Age'].median())
    # encode categorical
    cat_cols = [c for c in ['Sex', 'Embarked'] if c in X.columns]
    encoder = None
    if cat_cols:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(X[cat_cols])
        enc_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), enc_df], axis=1)
    return X, y, encoder


def train_regression(X, y, output_dir='models', random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    mae = mean_absolute_error(y_val, preds)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE:  {mae:.4f}")
    joblib.dump(model, os.path.join(output_dir, 'regression_baseline.pkl'))
    return model, rmse, mae


def main(train_csv, output_dir):
    print("Loading data:", train_csv)
    df = load_data(train_csv)
    df = basic_cleaning(df)
    if 'Fare' not in df.columns:
        raise ValueError("train.csv must contain a 'Fare' column for regression target")
    X, y, encoder = build_features_for_regression(df)
    print("Feature matrix shape:", X.shape)
    model, rmse, mae = train_regression(X, y, output_dir=output_dir)
    if encoder is not None:
        joblib.dump(encoder, os.path.join(output_dir, 'regression_encoder.pkl'))
    print("Done. Regression model saved to", output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--output-dir', default='models')
    args = parser.parse_args()
    main(args.train_csv, args.output_dir)
