# Experiment Settings

## Data

- Dataset split: Fatty-Liver-public split into train/validation with 8:2 stratified random split.
- Independent test set: Fatty-Liver-private-test/Liver.
- Input size: 224 x 224 RGB. Grayscale ultrasound images are converted to 3-channel RGB tensors for ImageNet-pretrained models.

## Models

- ResNet18: ImageNet-pretrained backbone, final fully connected layer replaced by a 2-class classifier.
- MobileNetV3-Small: ImageNet-pretrained backbone, final classifier layer replaced by a 2-class classifier.
- EfficientNet-B0: ImageNet-pretrained backbone, final classifier layer replaced by a 2-class classifier.

## Hyperparameters

- Loss: weighted cross entropy. Class weights were computed from the training split.
- Optimizer: AdamW.
- Initial learning rate: 0.0001.
- Weight decay: 0.0001.
- Learning-rate schedule: CosineAnnealingLR with T_max equal to the number of epochs.
- Epochs: 30.
- Batch size: 32.
- Random seeds: 42, 43, 44.

## Training Environment

- Hardware: NVIDIA GeForce RTX 4090, 24 GB GPU memory.
- Software: Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA enabled.

## AI Model Usage Statement

This project used OpenAI Codex / GPT-5 as an AI programming assistant. The assistant helped inspect the dataset structure, build PyTorch training and evaluation scripts, run experiments on the AutoDL server, organize metrics, export Grad-CAM visualizations, and draft report-ready experiment notes. Model selection, result interpretation, and final report writing remain the responsibility of the student.
