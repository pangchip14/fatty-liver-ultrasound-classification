# Fatty Liver Ultrasound Classification

This repository contains the code and report assets for a B-mode ultrasound fatty liver binary classification assignment.

## Scope

- Models: ResNet18, MobileNetV3-Small, EfficientNet-B0
- Task: normal liver vs fatty liver classification
- Training split: `Fatty-Liver-public` stratified into train/validation with an 8:2 ratio
- Test split: `Fatty-Liver-private-test` used only for independent evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, paired two-sided t-tests
- Interpretability: Grad-CAM scripts are included

## Privacy Note

The private test dataset is not included. Model checkpoints, raw data, archives, and Grad-CAM images containing private test images are intentionally excluded from GitHub.

Uploaded result assets include aggregate metrics, loss curves, topology diagrams, and the Markdown report.

## Main Files

- `train.py`: model training
- `evaluate.py`: independent test-set evaluation
- `gradcam.py`: Grad-CAM generation
- `summarize_results.py`: metric aggregation and t-test summary
- `export_report_assets.py`: report figures and settings export
- `脂肪肝二分类实验报告.md`: final Markdown report
- `summary/`: non-sensitive report assets and result tables

## Reproduce

Place the dataset zip under:

```text
data/ultrasoud-fatty-Liver-classification-data.zip
```

Generate splits:

```bash
python scripts/make_splits.py \
  --zip data/ultrasoud-fatty-Liver-classification-data.zip \
  --out-dir data \
  --seed 42 \
  --val-ratio 0.2
```

Run all experiments:

```bash
python run_experiments.py \
  --models resnet18 mobilenet_v3_small efficientnet_b0 \
  --seeds 42 43 44 \
  --epochs 30 \
  --batch-size 32 \
  --loss weighted_ce
```

