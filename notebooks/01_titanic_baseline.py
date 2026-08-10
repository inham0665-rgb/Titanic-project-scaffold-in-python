# %% [markdown]
# Titanic - Baseline notebook (py file that works as Jupyter cells)
# Follow the README to set up the environment, then run cells interactively.

# %%
import os
from src.data import load_data, basic_cleaning
from src.features import extract_features
from src.model import train_and_evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# %%
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

# %%
# Load and inspect
df = load_data(TRAIN_CSV)
print(df.shape)
print(df.head())

# %% [markdown]
# Basic cleaning

# %%
dfc = basic_cleaning(df)
print(dfc.isnull().sum())

# %% [markdown]
# Feature extraction

# %%
X, y, encoder = extract_features(dfc)
print("X shape:", X.shape)
print("Features:", list(X.columns))

# %% [markdown]
# Train baseline and evaluate

# %%
model, acc, roc = train_and_evaluate(X, y, output_dir="models")

# %% [markdown]
# Quick plot: survival by Pclass

# %%
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival rate by Pclass")
plt.show()

# %%
# Save notebook progress as a script or continue exploring
