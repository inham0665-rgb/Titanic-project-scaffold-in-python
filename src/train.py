"""
Simple training entrypoint:
python src/train.py --train-csv data/train.csv --output-dir models
"""
import argparse
from src.data import load_data, basic_cleaning
from src.features import extract_features
from src.model import train_and_evaluate
import joblib
import os

def main(train_csv, output_dir):
    print("Loading data:", train_csv)
    df = load_data(train_csv)
    df = basic_cleaning(df)
    X, y, encoder = extract_features(df)
    print("Feature matrix shape:", X.shape)
    model, acc, roc = train_and_evaluate(X, y, output_dir=output_dir)
    # Save encoder if present
    if encoder is not None:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(encoder, os.path.join(output_dir, 'encoder.pkl'))
    print("Done. Model and encoder (if any) saved to", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    main(args.train_csv, args.output_dir)
