"""
Clean the raw data and split into train/test.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from features import RAW_INPUT_COLUMNS, TARGET  # noqa: E402


def main():
    params = yaml.safe_load(open("params.yaml"))
    dp, pp = params["data"], params["preprocess"]
    out_dir = Path(dp["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dp["raw_path"])
    n_raw = len(df)


    df = df.drop_duplicates(subset="PassengerId")
    df = df[df[TARGET].isin([0, 1])]
    df["Sex"] = df["Sex"].str.strip().str.lower()
    df["Embarked"] = df["Embarked"].str.strip()

    train_df, test_df = train_test_split(
        df, test_size=dp["test_size"], random_state=dp["random_state"], stratify=df[TARGET]
    )


    fare_cap = float(train_df["Fare"].quantile(pp["fare_cap_quantile"]))
    n_capped = int((train_df["Fare"] > fare_cap).sum())
    train_df = train_df.assign(Fare=train_df["Fare"].clip(upper=fare_cap))
    test_df = test_df.assign(Fare=test_df["Fare"].clip(upper=fare_cap))

    cols = ["PassengerId"] + RAW_INPUT_COLUMNS + [TARGET]
    train_df[cols].to_csv(out_dir / "train.csv", index=False)
    test_df[cols].to_csv(out_dir / "test.csv", index=False)

    stats = {
        "raw_rows": n_raw,
        "clean_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "fare_cap": round(fare_cap, 2),
        "fares_capped_in_train": n_capped,
        "missing_age_pct": round(df["Age"].isna().mean() * 100, 1),
        "missing_cabin_pct": round(df["Cabin"].isna().mean() * 100, 1),
    }
    json.dump(stats, open(out_dir / "preprocess_stats.json", "w"), indent=2)
    print("Preprocessing done:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()