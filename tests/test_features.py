import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from features import MODEL_FEATURES, engineer_features, extract_title  # noqa: E402


def test_extract_title():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Rothes, the Countess. of (Lucy)") == "Rare"
    assert extract_title("Smith, Mlle. Anne") == "Miss"
    assert extract_title("no title here") == "Unknown"


def test_engineer_features_handles_missing_values():
    raw = pd.DataFrame([{
        "Pclass": 3, "Name": "Doe, Mr. John", "Sex": "male", "Age": None,
        "SibSp": 1, "Parch": 2, "Fare": 7.25, "Cabin": None, "Embarked": None,
    }])
    out = engineer_features(raw)
    assert list(out.columns) == MODEL_FEATURES
    assert out.loc[0, "FamilySize"] == 4
    assert out.loc[0, "IsAlone"] == 0
    assert out.loc[0, "HasCabin"] == 0
    assert out.loc[0, "AgeGroup"] == "unknown"
    assert out.loc[0, "Title"] == "Mr"