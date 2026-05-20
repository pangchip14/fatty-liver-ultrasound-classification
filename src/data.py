from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class ImageStats:
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_transform(image_size: int, train: bool, stats: ImageStats = ImageStats()):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                transforms.Normalize(stats.mean, stats.std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(stats.mean, stats.std),
        ]
    )


class FattyLiverDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_size: int, train: bool = False, limit: int | None = None):
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        if limit is not None:
            self.frame = self.frame.head(limit).copy()
        self.transform = build_transform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label, str(row["path"])


def class_counts(csv_path: str | Path) -> dict[int, int]:
    frame = pd.read_csv(csv_path)
    counts = frame["label"].value_counts().to_dict()
    return {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}


def class_weights(csv_path: str | Path) -> torch.Tensor:
    counts = class_counts(csv_path)
    total = counts[0] + counts[1]
    weights = [total / max(1, 2 * counts[0]), total / max(1, 2 * counts[1])]
    return torch.tensor(weights, dtype=torch.float32)
