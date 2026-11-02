"""
CI/CD pipeline in Python.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PY = sys.executable
WATCHED = [Path("data/raw"), Path("src"), Path("params.yaml")]


def step(name, cmd):
    print(f"\n{'=' * 60}\n>> {name}: {' '.join(cmd)}\n{'=' * 60}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nPIPELINE FAILED at step '{name}'. Nothing was deployed.")
        sys.exit(result.returncode)
    print(f"<< {name} finished in {time.time() - t0:.1f}s")


def deploy():
    print(f"\n{'=' * 60}\n>> deploy\n{'=' * 60}")
    import mlflow
    params = yaml.safe_load(open("params.yaml"))
    name = params["train"]["registered_model_name"]
    metrics = json.load(open("models/metrics.json"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.MlflowClient()
    version = str(metrics["model_version"])
    client.set_registered_model_alias(name, "champion", version)
    print(f"MLflow: {name} v{version} is now 'champion'")

    prod = Path("models/production")
    prod.mkdir(parents=True, exist_ok=True)
    shutil.copy("models/model.joblib", prod / "model.joblib")
    shutil.copy("models/metrics.json", prod / "metrics.json")
    print(f"Copied model to {prod}/")

    try:
        import git
        repo = git.Repo(".", search_parent_directories=True)
        tag = f"model-v{version}-{datetime.now():%Y%m%d-%H%M%S}"
        repo.create_tag(tag, message=f"{metrics['best_model']} acc={metrics['accuracy']} "
                                     f"auc={metrics['roc_auc']}")
        print(f"Git tag created: {tag}  (push with: git push --tags)")
    except Exception as e:  # not a git repo or no commits yet - deployment still succeeds
        print(f"Git tag skipped ({type(e).__name__}: {e})")


def run_once(skip_monitor=False):
    started = time.time()
    step("preprocess", [PY, "src/preprocess.py"])
    step("train", [PY, "src/train.py"])
    step("test", [PY, "-m", "pytest", "-q"])
    deploy()
    if not skip_monitor:
        step("monitor", [PY, "src/monitor.py", "--source", "normal"])
    print(f"\nPIPELINE SUCCEEDED in {time.time() - started:.0f}s")
    print("Restart the API to serve the new model:  uvicorn api.main:app --reload")


def fingerprint():
    h = hashlib.md5()
    for p in WATCHED:
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            if "__pycache__" not in str(f):
                h.update(str(f).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="re-run when data or code changes")
    ap.add_argument("--interval", type=int, default=10, help="seconds between checks in watch mode")
    ap.add_argument("--skip-monitor", action="store_true")
    args = ap.parse_args()

    if not args.watch:
        run_once(args.skip_monitor)
        return

    last = None
    print(f"Watching {', '.join(map(str, WATCHED))} for changes (Ctrl+C to stop)...")
    while True:
        current = fingerprint()
        if current != last:
            if last is not None:
                print("\nChange detected - re-running pipeline")
            try:
                run_once(args.skip_monitor)
            except SystemExit:
                print("Pipeline failed; waiting for the next change...")
            last = fingerprint()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()