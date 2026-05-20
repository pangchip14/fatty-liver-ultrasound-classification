from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def maybe_extract(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists():
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")


def collect_class_dir(root: Path) -> pd.DataFrame:
    records = []
    for label in [0, 1]:
        class_dir = root / str(label)
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                records.append({"path": str(path.resolve()), "label": label})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    args = parser.parse_args()

    data_root = args.out_dir / "extracted"
    split_dir = args.out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    maybe_extract(args.zip, data_root)

    public_root = data_root / "Fatty-Liver-public" / "Fatty-Liver"
    test_root = data_root / "Fatty-Liver-private-test" / "Liver"
    public_df = collect_class_dir(public_root)
    test_df = collect_class_dir(test_root)

    train_df, val_df = train_test_split(
        public_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=public_df["label"],
        shuffle=True,
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    stats = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train": train_df["label"].value_counts().sort_index().to_dict(),
        "val": val_df["label"].value_counts().sort_index().to_dict(),
        "test": test_df["label"].value_counts().sort_index().to_dict(),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    (split_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
