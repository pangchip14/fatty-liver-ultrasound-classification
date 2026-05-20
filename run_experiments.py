from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    parser.add_argument("--epochs", default="30")
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--loss", default="weighted_ce")
    parser.add_argument("--project-dir", default="/root/autodl-tmp/fatty_liver_project", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    for model in args.models:
        for seed in args.seeds:
            cmd = [
                sys.executable,
                str(args.project_dir / "train.py"),
                "--model",
                model,
                "--seed",
                seed,
                "--epochs",
                args.epochs,
                "--batch-size",
                args.batch_size,
                "--loss",
                args.loss,
            ]
            if args.no_pretrained:
                cmd.append("--no-pretrained")
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
