


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

