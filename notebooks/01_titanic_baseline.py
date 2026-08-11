# %% [markdown]
# Titanic - Baseline notebook with plotting cells
# This notebook adds ready-to-run plotting cells for recommended charts (EDA & evaluation).

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import load_data, basic_cleaning
from src.features import extract_features
from src.model import train_and_evaluate
from sklearn.metrics import confusion_matrix, roc_curve, auc

sns.set(style="whitegrid")

# %%
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

# %%
# Load and inspect
df = load_data(TRAIN_CSV)
print("Raw shape:", df.shape)
print(df.head())

# %% [markdown]
# Basic cleaning
# %%
df = basic_cleaning(df)
print("After basic cleaning, nulls:\n", df.isnull().sum())

# %% [markdown]
# Feature engineering (local to notebook)
# - Title extraction from Name
# - FamilySize and IsAlone
# - Age and Fare bins (optional)

# %%
# Title extraction
if 'Name' in df.columns:
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
    # Group rare titles
    title_counts = df['Title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
else:
    df['Title'] = 'Unknown'

# Family size
df['FamilySize'] = df.get('SibSp', 0) + df.get('Parch', 0) + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Age bins (coarse)
if 'Age' in df.columns:
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,20,40,60,120], labels=['Child','Teen','Adult','Mature','Senior'])
else:
    df['AgeBin'] = 'Unknown'

# Fare bins
if 'Fare' in df.columns:
    df['FareBin'] = pd.qcut(df['Fare'].fillna(0)+1, 4, labels=['Low','Med','High','VeryHigh'])
else:
    df['FareBin'] = 'Unknown'

print(df[['Title','FamilySize','IsAlone','AgeBin','FareBin']].head())

# %% [markdown]
# Chart 1: Survival rate by Pclass
# %%
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival rate by Pclass')
plt.ylabel('Survival rate')
plt.show()

# %% [markdown]
# Chart 2: Survival rate by Sex
# %%
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival rate by Sex')
plt.show()

# %% [markdown]
# Chart 3: Age distribution (survived vs not)
# %%
plt.figure(figsize=(8,4))
if 'Age' in df.columns:
    sns.kdeplot(df.loc[df['Survived']==1,'Age'].dropna(), label='Survived', shade=True)
    sns.kdeplot(df.loc[df['Survived']==0,'Age'].dropna(), label='Did not survive', shade=True)
    plt.title('Age distribution by survival')
    plt.xlabel('Age')
    plt.legend()
    plt.show()
else:
    print('Age column missing.')

# %% [markdown]
# Chart 4: Fare distribution by survival (boxplot)
# %%
plt.figure(figsize=(8,4))
if 'Fare' in df.columns:
    sns.boxplot(x='Survived', y='Fare', data=df)
    plt.title('Fare distribution by survival')
    plt.show()
else:
    print('Fare column missing.')

# %% [markdown]
# Chart 5: Survival by Embarked
# %%
plt.figure(figsize=(6,4))
if 'Embarked' in df.columns:
    sns.barplot(x='Embarked', y='Survived', data=df)
    plt.title('Survival rate by Embarked')
    plt.show()
else:
    print('Embarked column missing.')

# %% [markdown]
# Chart 6: Family size vs survival
# %%
plt.figure(figsize=(8,4))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Survival rate by FamilySize')
plt.show()

# %% [markdown]
# Chart 7: Title vs survival
# %%
plt.figure(figsize=(8,4))
sns.barplot(x='Title', y='Survived', data=df, order=sorted(df['Title'].unique()))
plt.title('Survival rate by Title')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# Chart 8: Age vs Fare colored by survival (scatter)
# Use a sample for speed
# %%
plt.figure(figsize=(8,6))
if 'Age' in df.columns and 'Fare' in df.columns:
    sample = df.sample(frac=0.5, random_state=42) if len(df)>1000 else df
    sns.scatterplot(x='Age', y='Fare', hue='Survived', data=sample, alpha=0.7, palette='Set1')
    plt.title('Age vs Fare colored by Survival')
    plt.show()
else:
    print('Age or Fare column missing.')

# %% [markdown]
# Chart 9: Correlation heatmap (numeric features)
# %%
plt.figure(figsize=(10,8))
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation heatmap (numeric features)')
    plt.show()
else:
    print('No numeric columns for correlation heatmap.')

# %% [markdown]
# Train baseline model (uses extract_features to build X)
# NOTE: extract_features will drop Survived from df when creating X/y
# %%
# Ensure we use the cleaned df copy with engineered features where applicable
# For modelling we'll drop non-numeric columns (Name, Ticket, Cabin, etc.) inside extract_features
X, y, encoder = extract_features(df)
print('Feature matrix shape:', X.shape)
model, acc, roc = train_and_evaluate(X, y, output_dir='models')

# %% [markdown]
# Chart 10: Feature importance (RandomForest)
# %%
plt.figure(figsize=(8,6))
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feat_names = X.columns
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=imp_df)
    plt.title('Feature importances (RandomForest)')
    plt.show()
else:
    print('Model does not expose feature_importances_')

# %% [markdown]
# Chart 11: Confusion matrix
# %%
plt.figure(figsize=(6,5))
# Get validation split predictions by reusing train_test_split logic (do a quick split)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
val_preds = model.predict(X_val)
cm = confusion_matrix(y_val, val_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# Chart 12: ROC curve & AUC
# %%
plt.figure(figsize=(6,5))
if hasattr(model, 'predict_proba'):
    y_prob = model.predict_proba(X_val)[:,1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
else:
    print('Model does not provide predict_proba; cannot plot ROC curve.')

# %%
print('Notebook plotting cells complete.')
