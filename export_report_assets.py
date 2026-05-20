from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import ttest_rel


RUN_RE = re.compile(r"(?P<model>.+)_seed(?P<seed>\d+)_(?P<stamp>\d{8}-\d{6})$")


DISPLAY_NAMES = {
    "resnet18": "ResNet18",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}


def parse_run_name(run_name: str) -> tuple[str, int]:
    match = RUN_RE.match(run_name)
    if not match:
        raise ValueError(f"Cannot parse run name: {run_name}")
    return match.group("model"), int(match.group("seed"))


def draw_flowchart(title: str, blocks: list[str], output: Path) -> None:
    fig_h = max(5.0, 0.72 * len(blocks) + 1.4)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(blocks) + 1.5)
    ax.axis("off")
    ax.text(5, len(blocks) + 1.05, title, ha="center", va="center", fontsize=16, fontweight="bold")

    y = len(blocks)
    for idx, text in enumerate(blocks):
        color = "#edf2ff" if idx not in (0, len(blocks) - 1) else "#e8f5e9"
        box = FancyBboxPatch(
            (1.1, y - 0.36),
            7.8,
            0.58,
            boxstyle="round,pad=0.018,rounding_size=0.045",
            linewidth=1.2,
            edgecolor="#2f3a4a",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(5, y - 0.07, text, ha="center", va="center", fontsize=10.5)
        if idx < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(5, y - 0.74),
                xytext=(5, y - 0.39),
                arrowprops={"arrowstyle": "->", "linewidth": 1.1, "color": "#2f3a4a"},
            )
        y -= 1

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_topologies(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    diagrams = {
        "resnet18": [
            "Input ultrasound image: 3 x 224 x 224",
            "7 x 7 Conv, 64 channels, stride 2 + BN + ReLU",
            "3 x 3 MaxPool, stride 2",
            "Layer1: BasicBlock x 2, 64 channels",
            "Layer2: BasicBlock x 2, 128 channels, downsample",
            "Layer3: BasicBlock x 2, 256 channels, downsample",
            "Layer4: BasicBlock x 2, 512 channels, downsample",
            "Global Average Pooling",
            "Fully Connected: 512 -> 2 classes",
        ],
        "mobilenet_v3_small": [
            "Input ultrasound image: 3 x 224 x 224",
            "3 x 3 Conv stem, 16 channels + Hardswish",
            "Inverted Residual Blocks with depthwise conv",
            "Squeeze-and-Excitation in selected blocks",
            "Progressive channel expansion: 16 -> 24 -> 40 -> 48 -> 96",
            "1 x 1 Conv head + Hardswish",
            "Global Average Pooling",
            "Classifier: Linear + Hardswish + Dropout",
            "Fully Connected: hidden -> 2 classes",
        ],
        "efficientnet_b0": [
            "Input ultrasound image: 3 x 224 x 224",
            "3 x 3 Conv stem, 32 channels + SiLU",
            "MBConv1 stage: 16 channels",
            "MBConv6 stages: 24, 40, 80, 112, 192, 320 channels",
            "Squeeze-and-Excitation inside MBConv blocks",
            "Compound-scaled depth/width baseline B0",
            "1 x 1 Conv head, 1280 channels + SiLU",
            "Global Average Pooling + Dropout",
            "Fully Connected: 1280 -> 2 classes",
        ],
    }
    for model, blocks in diagrams.items():
        draw_flowchart(DISPLAY_NAMES[model], blocks, out_dir / f"{model}_topology.png")


def export_loss_curves(outputs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    histories = []
    for run_dir in sorted(outputs_dir.iterdir()):
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        model, seed = parse_run_name(run_dir.name)
        history = pd.read_json(history_path)
        history["model"] = model
        history["seed"] = seed
        history["run"] = run_dir.name
        histories.append(history)
    if not histories:
        raise FileNotFoundError(f"No history.json files under {outputs_dir}")
    frame = pd.concat(histories, ignore_index=True)

    for model, model_df in frame.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for loss_name, color in [("train_loss", "#1f77b4"), ("val_loss", "#d62728")]:
            pivot = model_df.pivot_table(index="epoch", columns="seed", values=loss_name)
            mean = pivot.mean(axis=1)
            std = pivot.std(axis=1)
            x = mean.index.to_numpy()
            mean_values = mean.to_numpy(dtype=float)
            std_values = std.fillna(0).to_numpy(dtype=float)
            ax.plot(x, mean_values, label=loss_name.replace("_", " "), color=color, linewidth=2)
            ax.fill_between(x, mean_values - std_values, mean_values + std_values, color=color, alpha=0.16)
        ax.set_title(f"{DISPLAY_NAMES.get(model, model)} loss curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_loss_curve.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, (model, model_df) in zip(axes, sorted(frame.groupby("model"))):
        for loss_name, color in [("train_loss", "#1f77b4"), ("val_loss", "#d62728")]:
            pivot = model_df.pivot_table(index="epoch", columns="seed", values=loss_name)
            ax.plot(pivot.index, pivot.mean(axis=1), label=loss_name.replace("_", " "), color=color, linewidth=2)
        ax.set_title(DISPLAY_NAMES.get(model, model))
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[-1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "all_models_loss_curves.png", dpi=220)
    plt.close(fig)


def export_seed_t_tests(summary_dir: Path) -> None:
    run_df = pd.read_csv(summary_dir / "test_metrics_by_run.csv")
    model_summary = pd.read_csv(summary_dir / "test_metrics_by_model.csv")
    best_model = str(model_summary.sort_values(["f1_mean", "accuracy_mean"], ascending=False).iloc[0]["model"])
    metrics = ["accuracy", "precision", "recall", "f1"]
    rows = []
    best_df = run_df[run_df["model"] == best_model].sort_values("seed")
    for model in sorted(run_df["model"].unique()):
        other_df = run_df[run_df["model"] == model].sort_values("seed")
        if best_df["seed"].tolist() != other_df["seed"].tolist():
            raise ValueError(f"Seeds do not align for {best_model} and {model}")
        row = {"best_model": best_model, "compared_model": model, "seeds": ",".join(map(str, best_df["seed"]))}
        for metric in metrics:
            stat = ttest_rel(best_df[metric].to_numpy(), other_df[metric].to_numpy())
            row[f"{metric}_t"] = float(stat.statistic) if not np.isnan(stat.statistic) else 0.0
            row[f"{metric}_p"] = float(stat.pvalue) if not np.isnan(stat.pvalue) else 1.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(summary_dir / "paired_t_tests_by_seed.csv", index=False)


def export_settings(summary_dir: Path, outputs_dir: Path) -> None:
    first_config = sorted(outputs_dir.glob("*/config.json"))[0].read_text(encoding="utf-8")
    config = json.loads(first_config)
    lines = [
        "# Experiment Settings",
        "",
        "## Data",
        "",
        "- Dataset split: Fatty-Liver-public split into train/validation with 8:2 stratified random split.",
        "- Independent test set: Fatty-Liver-private-test/Liver.",
        "- Input size: 224 x 224 RGB. Grayscale ultrasound images are converted to 3-channel RGB tensors for ImageNet-pretrained models.",
        "",
        "## Models",
        "",
        "- ResNet18: ImageNet-pretrained backbone, final fully connected layer replaced by a 2-class classifier.",
        "- MobileNetV3-Small: ImageNet-pretrained backbone, final classifier layer replaced by a 2-class classifier.",
        "- EfficientNet-B0: ImageNet-pretrained backbone, final classifier layer replaced by a 2-class classifier.",
        "",
        "## Hyperparameters",
        "",
        f"- Loss: weighted cross entropy. Class weights were computed from the training split.",
        f"- Optimizer: AdamW.",
        f"- Initial learning rate: {config.get('lr')}.",
        f"- Weight decay: {config.get('weight_decay')}.",
        f"- Learning-rate schedule: CosineAnnealingLR with T_max equal to the number of epochs.",
        f"- Epochs: {config.get('epochs')}.",
        f"- Batch size: {config.get('batch_size')}.",
        f"- Random seeds: 42, 43, 44.",
        "",
        "## Training Environment",
        "",
        "- Hardware: NVIDIA GeForce RTX 4090, 24 GB GPU memory.",
        f"- Software: Python 3.12, PyTorch {config.get('torch')}, torchvision 0.23.0+cu128, CUDA enabled.",
        "",
        "## AI Model Usage Statement",
        "",
        "This project used OpenAI Codex / GPT-5 as an AI programming assistant. The assistant helped inspect the dataset structure, build PyTorch training and evaluation scripts, run experiments on the AutoDL server, organize metrics, export Grad-CAM visualizations, and draft report-ready experiment notes. Model selection, result interpretation, and final report writing remain the responsibility of the student.",
        "",
    ]
    (summary_dir / "experiment_settings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="/root/autodl-tmp/fatty_liver_project/outputs", type=Path)
    parser.add_argument("--summary-dir", default="/root/autodl-tmp/fatty_liver_project/summary", type=Path)
    args = parser.parse_args()

    export_loss_curves(args.outputs_dir, args.summary_dir / "loss_curves")
    export_topologies(args.summary_dir / "topologies")
    export_seed_t_tests(args.summary_dir)
    export_settings(args.summary_dir, args.outputs_dir)
    print(f"Report assets exported to {args.summary_dir}")


if __name__ == "__main__":
    main()
