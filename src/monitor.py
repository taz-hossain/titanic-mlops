"""
Monitor data drift and model performance with Evidently AI.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset

sys.path.append(str(Path(__file__).resolve().parent))
from features import (BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,  # noqa: E402
                      TARGET, engineer_features)

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


def simulate_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Fake a shift in who is buying tickets: older, more 3rd class, cheaper fares, more Queenstown."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["Age"] = d["Age"] + rng.normal(12, 5, len(d))
    d["Pclass"] = np.where(rng.random(len(d)) < 0.4, 3, d["Pclass"])
    d["Fare"] = d["Fare"] * rng.uniform(0.3, 0.7, len(d))
    d["Embarked"] = np.where(rng.random(len(d)) < 0.35, "Q", d["Embarked"])
    return d


def score(model, raw: pd.DataFrame) -> pd.DataFrame:
    feats = engineer_features(raw)
    feats["prediction_proba"] = model.predict_proba(feats[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES])[:, 1]
    feats["prediction"] = (feats["prediction_proba"] >= 0.5).astype(int)
    if TARGET in raw:
        feats[TARGET] = raw[TARGET].values
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["normal", "drifted", "api"], default="normal")
    ap.add_argument("--fail-on-alert", action="store_true",
                    help="exit with code 1 if an alert fires (useful in CI)")
    args = ap.parse_args()

    params = yaml.safe_load(open("params.yaml"))
    mp = params["monitoring"]
    proc = Path(params["data"]["processed_dir"])
    model = joblib.load("models/model.joblib")

    test_raw = pd.read_csv(proc / "test.csv")
    reference = score(model, pd.read_csv(proc / "train.csv"))   # drift baseline
    perf_reference = score(model, test_raw)                      # performance baseline
    if args.source == "normal":
        current = score(model, test_raw.sample(frac=1.0, replace=True, random_state=1))
    elif args.source == "drifted":
        current = score(model, simulate_drift(test_raw))
    else:
        log = Path("logs/predictions.csv")
        if not log.exists():
            sys.exit("No API logs yet. Send some requests to the API first.")
        current = score(model, pd.read_csv(log))
    has_labels = TARGET in current

    numeric = NUMERIC_FEATURES + ["prediction_proba"]
    categorical = CATEGORICAL_FEATURES + BINARY_FEATURES + ["prediction"]
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

    def to_dataset(df, with_labels):
        cols = numeric + categorical + ([TARGET] if with_labels else [])
        definition = DataDefinition(
            numerical_columns=numeric,
            categorical_columns=categorical + ([TARGET] if with_labels else []),
            classification=[BinaryClassification(target=TARGET, prediction_labels="prediction",
                                                 prediction_probas="prediction_proba")]
            if with_labels else None,
        )
        return Dataset.from_pandas(df[cols], data_definition=definition)

    ref_ds = to_dataset(reference, False)
    cur_ds = to_dataset(current, False)


    drift_report = Report([DataDriftPreset(columns=feature_cols + ["prediction"]), DataSummaryPreset()])
    drift_snapshot = drift_report.run(current_data=cur_ds, reference_data=ref_ds)
    drift_html = REPORTS / f"data_drift_{args.source}.html"
    drift_snapshot.save_html(str(drift_html))
    drift_json = drift_snapshot.dict()

    drifted_cols, drift_share = [], 0.0
    for m in drift_json["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("DriftedColumnsCount"):
            drift_share = float(m["value"]["share"])
        elif cfg.get("type", "").endswith("ValueDrift"):
            # p-value tests (K-S, chi-square, Z) flag drift when value < threshold;
            # distance tests (Wasserstein, Jensen-Shannon) flag drift when value >= threshold
            v, t = m["value"], cfg["threshold"]
            if ("p_value" in cfg["method"] and v < t) or ("p_value" not in cfg["method"] and v >= t):
                drifted_cols.append(cfg["column"])
    n_cols = len(feature_cols) + 1

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drift_share": round(drift_share, 3),
        "drifted_columns": drifted_cols,
        "reports": [str(drift_html)],
        "alerts": [],
    }
    if drift_share > mp["drift_share_threshold"]:
        summary["alerts"].append(
            f"DATA DRIFT: {len(drifted_cols)}/{n_cols} columns drifted "
            f"({drift_share:.0%} > {mp['drift_share_threshold']:.0%}): {drifted_cols}")


    if has_labels:
        perf_report = Report([ClassificationPreset()])
        perf_snapshot = perf_report.run(current_data=to_dataset(current, True),
                                        reference_data=to_dataset(perf_reference, True))
        perf_html = REPORTS / f"model_performance_{args.source}.html"
        perf_snapshot.save_html(str(perf_html))
        summary["reports"].append(str(perf_html))

        ref_acc = float((perf_reference["prediction"] == perf_reference[TARGET]).mean())
        cur_acc = float((current["prediction"] == current[TARGET]).mean())
        summary.update({"reference_accuracy": round(ref_acc, 4), "current_accuracy": round(cur_acc, 4),
                        "accuracy_drop": round(ref_acc - cur_acc, 4)})
        if ref_acc - cur_acc > mp["accuracy_drop_threshold"]:
            summary["alerts"].append(
                f"PERFORMANCE DROP: accuracy {ref_acc:.3f} -> {cur_acc:.3f} "
                f"(drop {ref_acc - cur_acc:.3f} > {mp['accuracy_drop_threshold']})")

    out = REPORTS / f"monitoring_summary_{args.source}.json"
    json.dump(summary, open(out, "w"), indent=2)

    print(json.dumps(summary, indent=2))
    if summary["alerts"]:
        print("\n" + "!" * 60)
        for a in summary["alerts"]:
            print("ALERT:", a)
        print("Recommended action: investigate the drift report and retrain (python run_pipeline.py)")
        print("!" * 60)
        if args.fail_on_alert:
            sys.exit(1)
    else:
        print("\nNo alerts. Model and data look stable.")


if __name__ == "__main__":
    main()