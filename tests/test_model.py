"""Model quality tests to act as the quality gate in CI/CD."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from features import TARGET, engineer_features  # noqa: E402

PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))


@pytest.fixture(scope="module")
def model():
    path = ROOT / "models" / "model.joblib"
    if not path.exists():
        pytest.skip("No trained model - run python src/train.py first")
    return joblib.load(path)


def test_model_meets_quality_gate():
    metrics = json.load(open(ROOT / "models" / "metrics.json"))
    gate = PARAMS["quality_gate"]
    assert metrics["accuracy"] >= gate["min_accuracy"], metrics
    assert metrics["roc_auc"] >= gate["min_roc_auc"], metrics


def test_predictions_are_valid_probabilities(model):
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    proba = model.predict_proba(engineer_features(test))[:, 1]
    assert len(proba) == len(test)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_learned_known_pattern(model):
    """Sanity check: a 1st-class woman should be far more likely to survive than a 3rd-class man."""
    rows = pd.DataFrame([
        {"Pclass": 1, "Name": "A, Mrs. B", "Sex": "female", "Age": 30, "SibSp": 1, "Parch": 0,
         "Fare": 80.0, "Cabin": "C85", "Embarked": "C"},
        {"Pclass": 3, "Name": "A, Mr. B", "Sex": "male", "Age": 30, "SibSp": 0, "Parch": 0,
         "Fare": 7.25, "Cabin": None, "Embarked": "S"},
    ])
    p_woman, p_man = model.predict_proba(engineer_features(rows))[:, 1]
    assert p_woman > 0.7
    assert p_man < 0.3


def test_no_target_leakage():
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    assert TARGET not in engineer_features(test).columns