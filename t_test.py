from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


def read_correct(path: Path):
    frame = pd.read_csv(path)
    frame = frame.sort_values("path")
    return (frame["label"].to_numpy() == frame["pred"].to_numpy()).astype(float), frame["path"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", required=True, type=Path, help="Predictions CSV for the best model.")
    parser.add_argument("--others", required=True, type=Path, nargs="+", help="Prediction CSV files to compare.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    best_correct, best_paths = read_correct(args.best)
    results = []
    for other in args.others:
        other_correct, other_paths = read_correct(other)
        if best_paths != other_paths:
            raise ValueError(f"Prediction paths do not align: {other}")
        stat = ttest_rel(best_correct, other_correct)
        results.append({"other": str(other), "t": float(stat.statistic), "p": float(stat.pvalue)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
