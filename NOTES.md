# Project Notes — Titanic Beginner ML Project

Quick summary
- Purpose: Predict passenger survival on the Titanic using a simple, reproducible Python pipeline.
- Baseline: RandomForest model with minimal preprocessing (age/fare imputation, one-hot encoding for Sex/Embarked).
- Location: src/ contains data, feature, training code. notebooks/ has an interactive script-style notebook.

How to run (short)
1. Clone the repo and enter it:
   git clone https://github.com/inham0665-rgb/Titanic-project-scaffold-in-python.git
   cd Titanic-project-scaffold-in-python
2. Create and activate a venv:
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Add data files:
   - Download `train.csv` and `test.csv` from Kaggle and place them into `data/`.
5. Train baseline:
   bash run_train.sh
   or
   python src/train.py --train-csv data/train.csv --output-dir models

Files of interest
- src/data.py — loading and minimal cleaning
- src/features.py — feature extraction and categorical encoding
- src/model.py — training, evaluation, and model saving
- src/train.py — CLI entrypoint
- notebooks/01_titanic_baseline.py — notebook-style script for interactive exploration
- models/ — output (created after running training)

Notes on reproducibility
- Random state fixed in train_and_evaluate (default 42) for reproducible splits & training.
- Save artefacts: models/baseline.pkl and models/encoder.pkl (if encoder exists).
- For full experiment tracking, consider adding MLflow or DVC.

Suggested immediate improvements (good first commits)
- Feature engineering:
  - Extract Title from Name (Mr/Miss/Mrs/Dr) and impute Age by title group.
  - Add FamilySize = SibSp + Parch + 1 and IsAlone flag.
  - Create Fare bins and Age bins.
- Validation:
  - Replace single holdout with stratified k-fold cross-validation.
  - Log metrics for each fold.
- Model tuning:
  - Try LogisticRegression, XGBoost/LightGBM, and simple hyperparameter search (GridSearchCV or RandomizedSearchCV).
- Packaging:
  - Add a small FastAPI inference endpoint and a Dockerfile for deployment.
- CI / Tests:
  - Add GitHub Actions to run lint/tests and a smoke training job on small sample data.

Helpful tips
- If you get sklearn warnings about OneHotEncoder(sparse=False), upgrade scikit-learn or accept the warning; the code uses get_feature_names_out().
- Keep data/ ignored in .gitignore to avoid committing the dataset; share only instructions to download.
- Use a small subset of the training set when iterating quickly to speed up development.

Links & resources
- Titanic dataset (Kaggle): https://www.kaggle.com/c/titanic/data
- Titanic tutorial (Kaggle Learn): https://www.kaggle.com/learn/intro-to-machine-learning
- pandas docs: https://pandas.pydata.org/docs/
- scikit-learn guide: https://scikit-learn.org/stable/tutorial/index.html
- Feature engineering ideas: https://www.kaggle.com/competitions/titanic/overview
- Model explainability basics (SHAP): https://shap.readthedocs.io/
- Experiment tracking (MLflow): https://mlflow.org/
- Data versioning (DVC): https://dvc.org/
- Quick ML deployment with FastAPI: https://fastapi.tiangolo.com/

Repository and contact
- Repo: https://github.com/inham0665-rgb/Titanic-project-scaffold-in-python
- Commit used for scaffold: https://github.com/inham0665-rgb/Titanic-project-scaffold-in-python/commit/209cb3c887d672dc949daa92809bc5f2b0ce63bf

License
- MIT (see LICENSE file)

Last updated: 2026-08-10
