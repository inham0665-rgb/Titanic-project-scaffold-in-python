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
