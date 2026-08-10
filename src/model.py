from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os

def train_and_evaluate(X, y, output_dir='models', random_state=42):
    """
    Train a baseline model (RandomForest) and evaluate.
    Saves model to output_dir/baseline.pkl
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = None
    try:
        probs = model.predict_proba(X_val)[:, 1]
    except Exception:
        pass
    acc = accuracy_score(y_val, preds)
    roc = roc_auc_score(y_val, probs) if probs is not None else None
    print("Validation accuracy: {:.4f}".format(acc))
    if roc is not None:
        print("Validation ROC AUC: {:.4f}".format(roc))
    print("Classification report:")
    print(classification_report(y_val, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, preds))
    joblib.dump(model, os.path.join(output_dir, 'baseline.pkl'))
    return model, acc, roc
