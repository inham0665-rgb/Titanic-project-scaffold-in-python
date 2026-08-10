import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV dataset from path.
    """
    return pd.read_csv(path)

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning:
    - Fill Age with median
    - Fill Embarked with mode
    - Drop Cabin (too many missing), Ticket (noisy), PassengerId (keep if needed)
    """
    df = df.copy()
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Drop Cabin and Ticket for baseline
    for col in ['Cabin', 'Ticket']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df
