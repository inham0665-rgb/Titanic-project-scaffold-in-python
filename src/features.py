import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def extract_features(df: pd.DataFrame, fit_encoder: OneHotEncoder = None):
    """
    Convert dataframe to numeric features for baseline:
    - Use Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    - Encode Sex and Embarked with one-hot
    Returns X (DataFrame), y (Series if present), encoder (fitted)
    """
    df = df.copy()
    y = None
    if 'Survived' in df.columns:
        y = df['Survived']
        df = df.drop(columns=['Survived'])
    # Keep relevant columns (if exist)
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    cols = [c for c in cols if c in df.columns]
    X = df[cols].copy()
    # Fill missing Fare
    if 'Fare' in X.columns:
        X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    # Categorical encoding
    cat_cols = [c for c in ['Sex', 'Embarked'] if c in X.columns]
    if cat_cols:
        encoder = fit_encoder
        if encoder is None:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(X[cat_cols])
        else:
            encoded = encoder.transform(X[cat_cols])
        enc_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), enc_df], axis=1)
    else:
        encoder = fit_encoder
    return X, y, encoder
