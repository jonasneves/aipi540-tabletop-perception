"""End-to-end build: fetch data, featurize, train all three models, export artifacts.

This satisfies the AIPI 540 rubric's expected `setup.py` entrypoint. It calls
the per-model scripts in `scripts/` which are independently runnable. Nothing
here is graded at runtime; this is the one-command reproduction path.

Usage:
    python setup.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: str) -> None:
    print(f"\n$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=ROOT)


def main() -> int:
    run("python3 scripts/make_dataset.py --out data/processed")
    run("python3 scripts/naive.py --train data/processed/train --test data/processed/test --out results/naive.json")
    run(
        "python3 scripts/classical.py "
        "--train data/processed/train --test data/processed/test "
        "--out results/classical.json --save-model models/classical.pkl"
    )
    run(
        "python3 scripts/train_dl.py "
        "--train data/processed/train --test data/processed/test "
        "--out-dir models --results-dir results --epochs 12 --skip-detector"
    )
    os.makedirs(ROOT / "public" / "models", exist_ok=True)
    os.makedirs(ROOT / "public" / "results", exist_ok=True)
    run("cp models/classifier.onnx public/models/")
    run("cp results/naive.json results/classical.json results/dl.json public/results/")
    print("\nSetup complete. Start the app with: python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
