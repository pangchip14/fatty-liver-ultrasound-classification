from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import FattyLiverDataset
from src.metrics import bootstrap_metric_summary, classification_metrics
from src.models import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", default="/root/autodl-tmp/fatty_liver_project/data/splits/test.csv", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--n-bootstrap", default=1000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    model = create_model(model_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = FattyLiverDataset(args.csv, image_size=args.image_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="evaluate"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            for path, label, pred, prob in zip(paths, labels.numpy(), preds, probs):
                rows.append({"path": path, "label": int(label), "pred": int(pred), "prob_fatty": float(prob)})

    pred_df = pd.DataFrame(rows)
    metrics = classification_metrics(pred_df["label"], pred_df["pred"])
    summary = bootstrap_metric_summary(
        pred_df["label"].to_numpy(),
        pred_df["pred"].to_numpy(),
        n_bootstrap=args.n_bootstrap,
    )
    result = {"checkpoint": str(args.checkpoint), "model": model_name, "metrics": metrics, "bootstrap": summary}

    output_dir = args.output_dir or args.checkpoint.parents[1] / "test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
