from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import ImageStats
from src.models import create_model, gradcam_target_layer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, class_index: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, class_index].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, logits.detach()


def overlay_heatmap(original_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.55 * original_rgb + 0.45 * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", default="/root/autodl-tmp/fatty_liver_project/data/splits/test.csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--max-images", default=8, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    model = create_model(model_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    cam_runner = GradCAM(model, gradcam_target_layer(model, model_name))

    stats = ImageStats()
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(stats.mean, stats.std),
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.csv).head(args.max_images)

    for idx, row in frame.iterrows():
        image = Image.open(row["path"]).convert("RGB")
        resized = image.resize((args.image_size, args.image_size))
        tensor = transform(image).unsqueeze(0).to(device)
        cam, logits = cam_runner(tensor, class_index=1)
        pred = int(logits.argmax(dim=1).item())
        original = np.array(resized)
        overlay = overlay_heatmap(original, cam)
        canvas = np.concatenate([original, overlay], axis=1)
        out = args.output_dir / f"{idx:03d}_label{int(row['label'])}_pred{pred}.jpg"
        Image.fromarray(canvas).save(out, quality=95)
        print(out)


if __name__ == "__main__":
    main()
