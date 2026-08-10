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

If you want, I can push this scaffold to a new GitHub repo for you or create a single notebook .ipynb file instead of the .py notebook.
