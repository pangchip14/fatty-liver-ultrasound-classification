from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import FattyLiverDataset, class_weights
from src.losses import FocalLoss
from src.metrics import classification_metrics
from src.models import create_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--data-dir", default="/root/autodl-tmp/fatty_liver_project/data", type=Path)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/fatty_liver_project/outputs", type=Path)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--loss", default="weighted_ce", choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--limit-train", default=None, type=int)
    parser.add_argument("--limit-val", default=None, type=int)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []

    iterator = tqdm(loader, leave=False, desc="train" if train else "val")
    for images, labels, _paths in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy_score(y_true, y_pred):.4f}")

    avg_loss = total_loss / len(loader.dataset)
    metrics = classification_metrics(y_true, y_pred)
    return avg_loss, metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    split_dir = args.data_dir / "splits"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Run scripts/make_splits.py first.")

    run_name = f"{args.model}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = args.output_dir / run_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = FattyLiverDataset(train_csv, image_size=args.image_size, train=True, limit=args.limit_train)
    val_ds = FattyLiverDataset(val_csv, image_size=args.image_size, train=False, limit=args.limit_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(args.model, pretrained=not args.no_pretrained).to(device)
    weights = class_weights(train_csv).to(device)
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = FocalLoss(alpha=weights, gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_f1 = -1.0
    history = []

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["device"] = str(device)
    config["torch"] = torch.__version__
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        checkpoint = {
            "model": model.state_dict(),
            "model_name": args.model,
            "epoch": epoch,
            "config": config,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(checkpoint, ckpt_dir / "best.pt")

    writer.close()
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Finished. Best validation F1: {best_f1:.4f}. Output: {out_dir}")


if __name__ == "__main__":
    main()
