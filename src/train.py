"""
Train, evaluate, and version models with MLflow.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from features import (BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,  # noqa: E402
                      TARGET, engineer_features)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),          # missing Age / Fare
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),  # reduce skew
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),   # missing Embarked
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])


CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8],
         "clf__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [2, 3]},
    ),
}


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }, pred


def save_plots(model, X, y, name):
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, display_labels=["Died", "Survived"])
    ax.set_title(f"Confusion matrix - {name}")
    p = MODELS_DIR / f"confusion_matrix_{name}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax)
    ax.set_title(f"ROC curve - {name}")
    p = MODELS_DIR / f"roc_curve_{name}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def feature_importance(model):
    names = model.named_steps["prep"].get_feature_names_out()
    clf = model.named_steps["clf"]
    vals = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    return (pd.Series(vals, index=names).sort_values(ascending=False).round(4))


def main():
    params = yaml.safe_load(open("params.yaml"))
    tp = params["train"]
    proc = Path(params["data"]["processed_dir"])

    train_df = pd.read_csv(proc / "train.csv")
    test_df = pd.read_csv(proc / "test.csv")
    X_train, y_train = engineer_features(train_df), train_df[TARGET]
    X_test, y_test = engineer_features(test_df), test_df[TARGET]

    mlflow.set_experiment(tp["experiment_name"])
    cv = StratifiedKFold(n_splits=tp["cv_folds"], shuffle=True, random_state=42)
    results = {}

    for name, (estimator, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            metrics, _ = evaluate(best, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_metric("cv_roc_auc", search.best_score_)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            mlflow.set_tag("data_version", _dvc_data_hash())
            for p in save_plots(best, X_test, y_test, name):
                mlflow.log_artifact(str(p), artifact_path="plots")
            fi = feature_importance(best)
            fi_path = MODELS_DIR / f"feature_importance_{name}.csv"
            fi.to_csv(fi_path, header=["importance"])
            mlflow.log_artifact(str(fi_path))
            mlflow.sklearn.log_model(
                best, name="model",
                signature=infer_signature(X_train, best.predict(X_train)),
                input_example=X_train.head(3),
                # We trained these models ourselves, so these types are safe to trust (MLflow 3.x safe serialization)
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
            results[name] = {"model": best, "metrics": metrics, "run_id": run.info.run_id,
                             "cv_roc_auc": search.best_score_, "params": search.best_params_}
            print(f"{name:22s} cv_auc={search.best_score_:.3f} " +
                  " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    best = results[best_name]
    print(f"\nBest model (by CV ROC-AUC): {best_name}  cv_auc={best['cv_roc_auc']:.3f}  "
          f"test_auc={best['metrics']['roc_auc']:.3f}")

    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", tp["registered_model_name"])
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(tp["registered_model_name"], "candidate", mv.version)
    print(f"Registered {tp['registered_model_name']} v{mv.version} (alias: candidate)")


    joblib.dump(best["model"], MODELS_DIR / "model.joblib")
    metrics_out = {
        "best_model": best_name,
        "model_version": int(mv.version),
        "run_id": best["run_id"],
        "cv_roc_auc": round(best["cv_roc_auc"], 4),
        **{k: round(v, 4) for k, v in best["metrics"].items()},
        "all_models": {n: {k: round(v, 4) for k, v in r["metrics"].items()}
                       for n, r in results.items()},
    }
    json.dump(metrics_out, open(MODELS_DIR / "metrics.json", "w"), indent=2)

    _, pred = evaluate(best["model"], X_test, y_test)
    print("\nClassification report (test set):")
    print(classification_report(y_test, pred, target_names=["Died", "Survived"]))
    print("Top 10 features:")
    print(feature_importance(best["model"]).head(10))


def _dvc_data_hash():
    """Read the DVC md5 of the raw data so each MLflow run records which data it used."""
    dvc_file = Path("data/raw/titanic.csv.dvc")
    if dvc_file.exists():
        meta = yaml.safe_load(open(dvc_file))
        return meta["outs"][0].get("md5", "unknown")
    return "untracked"


if __name__ == "__main__":
    main()