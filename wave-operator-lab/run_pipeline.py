from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str, models: str) -> None:
    command = [sys.executable, str(ROOT / script), "--models", models]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete WaveOperator Lab experiment.")
    parser.add_argument("--models", default="fno,cnn")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    if not args.skip_training:
        run("train.py", args.models)
    run("evaluate.py", args.models)


if __name__ == "__main__":
    main()
