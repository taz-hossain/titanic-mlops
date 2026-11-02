"""Feature engineering shared by training AND the API."""
import numpy as np
import pandas as pd

# Raw columns the model needs. The API accepts exactly these.
RAW_INPUT_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "AgeGroup"]
BINARY_FEATURES = ["IsAlone", "HasCabin"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TARGET = "Survived"

_TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Capt": "Rare",
    "Sir": "Rare", "Lady": "Rare", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare", "the Countess": "Rare", "Countess": "Rare",
}


def extract_title(name: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr'. Unknown titles become 'Rare'."""
    if not isinstance(name, str) or "," not in name or "." not in name:
        return "Unknown"
    title = name.split(",", 1)[1].split(".", 1)[0].strip()
    return _TITLE_MAP.get(title, "Rare")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw passenger rows into model-ready features."""
    out = pd.DataFrame(index=df.index)
    out["Pclass"] = df["Pclass"].astype(str)
    out["Sex"] = df["Sex"].astype(str).str.lower()
    out["Embarked"] = df["Embarked"]
    out["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    out["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    out["SibSp"] = df["SibSp"].astype(int)
    out["Parch"] = df["Parch"].astype(int)

    out["Title"] = df["Name"].apply(extract_title)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["HasCabin"] = df["Cabin"].notna().astype(int) if "Cabin" in df else 0
    out["AgeGroup"] = pd.cut(
        out["Age"], bins=[0, 12, 18, 35, 60, np.inf],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype(str).replace("nan", "unknown")
    return out[MODEL_FEATURES]