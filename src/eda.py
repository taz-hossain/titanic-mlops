"""
Exploratory Data Analysis.
Prints a data summary and saves charts to reports/figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml"))
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(params["data"]["raw_path"])

print("=== Shape ===")
print(df.shape)
print("\n=== Data types ===")
print(df.dtypes)
print("\n=== Missing values ===")
missing = df.isna().sum()
print(pd.DataFrame({"missing": missing, "pct": (missing / len(df) * 100).round(1)})[missing > 0])
print("\n=== Numeric summary ===")
print(df.describe().round(2))
print("\n=== Overall survival rate ===")
print(f"{df['Survived'].mean():.1%}")

print("\n=== Survival rate by group ===")
for col in ["Sex", "Pclass", "Embarked"]:
    print(df.groupby(col)["Survived"].mean().round(3), "\n")

# Outlier check on Fare using the IQR rule
q1, q3 = df["Fare"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print(f"=== Fare outliers (IQR rule) ===\n{(df['Fare'] > upper).sum()} passengers paid > {upper:.2f}" f" (max = {df['Fare'].max():.2f})")


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Sex", "Pclass", "Embarked"]):
    df.groupby(col)["Survived"].mean().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(f"Survival rate by {col}")
    ax.set_ylabel("Survival rate")
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(FIG_DIR / "survival_by_group.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
df[df.Survived == 0]["Age"].plot(kind="hist", bins=30, alpha=0.6, label="Died", ax=ax)
df[df.Survived == 1]["Age"].plot(kind="hist", bins=30, alpha=0.6, label="Survived", ax=ax)
ax.set_title("Age distribution by survival")
ax.set_xlabel("Age")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "age_distribution.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
df.boxplot(column="Fare", by="Pclass", ax=ax)
ax.set_title("Fare by class (outliers visible above whiskers)")
plt.suptitle("")
plt.tight_layout()
plt.savefig(FIG_DIR / "fare_boxplot.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
corr = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("Correlation matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation.png", dpi=120)
plt.close()

print(f"\nCharts saved to {FIG_DIR}/")