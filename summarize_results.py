from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


RUN_RE = re.compile(r"(?P<model>.+)_seed(?P<seed>\d+)_(?P<stamp>\d{8}-\d{6})$")


def parse_run_name(path: Path) -> tuple[str, int]:
    match = RUN_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse run directory name: {path.name}")
    return match.group("model"), int(match.group("seed"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="/root/autodl-tmp/fatty_liver_project/outputs", type=Path)
    parser.add_argument("--summary-dir", default="/root/autodl-tmp/fatty_liver_project/summary", type=Path)
    args = parser.parse_args()

    rows = []
    prediction_files: dict[str, Path] = {}
    for run_dir in sorted(args.outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "test_eval" / "metrics.json"
        pred_path = run_dir / "test_eval" / "predictions.csv"
        if not metrics_path.exists() or not pred_path.exists():
            continue
        model, seed = parse_run_name(run_dir)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
        row = {"run": run_dir.name, "model": model, "seed": seed, **metrics}
        rows.append(row)
        prediction_files[run_dir.name] = pred_path

    if not rows:
        raise FileNotFoundError("No evaluated runs found. Run evaluate.py for each checkpoint first.")

    run_df = pd.DataFrame(rows).sort_values(["model", "seed"])
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    summary = run_df.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["f1_mean", "accuracy_mean"], ascending=False)

    best_model = str(summary.iloc[0]["model"])
    best_run = run_df.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
    best_pred = pd.read_csv(prediction_files[str(best_run["run"])]).sort_values("path")
    best_correct = (best_pred["label"].to_numpy() == best_pred["pred"].to_numpy()).astype(float)

    t_rows = []
    for model in sorted(run_df["model"].unique()):
        model_runs = run_df[run_df["model"] == model].sort_values(["f1", "accuracy"], ascending=False)
        representative = model_runs.iloc[0]
        pred = pd.read_csv(prediction_files[str(representative["run"])]).sort_values("path")
        if best_pred["path"].tolist() != pred["path"].tolist():
            raise ValueError(f"Prediction rows do not align for {representative['run']}")
        correct = (pred["label"].to_numpy() == pred["pred"].to_numpy()).astype(float)
        stat = ttest_rel(best_correct, correct)
        t_rows.append(
            {
                "best_run": str(best_run["run"]),
                "best_model": str(best_run["model"]),
                "compared_model": model,
                "compared_run": str(representative["run"]),
                "t": float(stat.statistic) if not np.isnan(stat.statistic) else 0.0,
                "p": float(stat.pvalue) if not np.isnan(stat.pvalue) else 1.0,
            }
        )

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(args.summary_dir / "test_metrics_by_run.csv", index=False)
    summary.to_csv(args.summary_dir / "test_metrics_by_model.csv", index=False)
    pd.DataFrame(t_rows).to_csv(args.summary_dir / "paired_t_tests.csv", index=False)
    report = {
        "best_model_by_mean_f1": best_model,
        "best_single_run_by_test_f1": str(best_run["run"]),
        "n_runs": int(len(run_df)),
    }
    (args.summary_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(summary.to_string(index=False))
    print(pd.DataFrame(t_rows).to_string(index=False))


if __name__ == "__main__":
    main()
