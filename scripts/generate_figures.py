#!/usr/bin/env python3
"""Generate clean paper figures using the exact same images as the pytorch-grad-cam notebook.

Source images: dog_labrador.jpg, cat_tabby.jpg, espresso.jpg, sports_car.jpg
(Wikimedia CC0, stored in bnnr/examples/integrations/assets/)

Run from bnnr/ repo root:
    PYTHONPATH=src python3 ../bnnr-research/scripts/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

REPO = Path(__file__).resolve().parents[2] / "bnnr"
ASSETS = REPO / "examples" / "integrations" / "assets"
OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

DEMO_FILES = ["dog_labrador.jpg", "cat_tabby.jpg", "espresso.jpg", "sports_car.jpg"]
DEMO_LABELS = ["Labrador retriever", "Tabby cat", "Espresso", "Sports car"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NormalizedModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def build_model() -> NormalizedModel:
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.eval()
    return NormalizedModel(backbone).eval()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def load_images() -> tuple[list[Image.Image], torch.Tensor]:
    pil_imgs = [Image.open(ASSETS / f).convert("RGB") for f in DEMO_FILES]
    tensors = torch.stack([preprocess(im) for im in pil_imgs])
    return pil_imgs, tensors


# ---------------------------------------------------------------------------
# GradCAM saliency
# ---------------------------------------------------------------------------

def compute_gradcam(model: NormalizedModel, tensors: torch.Tensor,
                    labels: list[int]) -> np.ndarray:
    from bnnr.xai import generate_saliency_maps
    model.eval()
    target_layers = [model.backbone.layer4[-1]]
    label_t = torch.tensor(labels)
    maps = generate_saliency_maps(model, tensors, label_t, target_layers, method="opticam")
    # Normalise each map to [0, 1]
    out = []
    for m in maps:
        lo, hi = float(m.min()), float(m.max())
        out.append(((m - lo) / (hi - lo + 1e-8)))
    return np.stack(out)   # (N, H, W)


# ---------------------------------------------------------------------------
# ICD / AICD helpers
# ---------------------------------------------------------------------------

def apply_icd(model: NormalizedModel, img_np: np.ndarray, label: int,
              tau: float = 0.50, tile_size: int = 8,
              fill: str = "gaussian_blur") -> np.ndarray:
    from bnnr.icd import ICD
    icd = ICD(
        model=model,
        target_layers=[model.backbone.layer4[-1]],
        threshold_percentile=tau * 100,
        probability=1.0,
        tile_size=tile_size,
        fill_strategy=fill,
        random_state=42,
    )
    return icd.apply_with_label(img_np, label)


def apply_aicd(model: NormalizedModel, img_np: np.ndarray, label: int,
               tau: float = 0.50, tile_size: int = 8,
               fill: str = "gaussian_blur") -> np.ndarray:
    from bnnr.icd import AICD
    aicd = AICD(
        model=model,
        target_layers=[model.backbone.layer4[-1]],
        threshold_percentile=tau * 100,
        probability=1.0,
        tile_size=tile_size,
        fill_strategy=fill,
        random_state=42,
    )
    return aicd.apply_with_label(img_np, label)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figure 1: comparison grid (original / Grad-CAM / ICD / AICD) × 4 images
# ---------------------------------------------------------------------------

def figure_comparison(model: NormalizedModel, tensors: torch.Tensor,
                      saliency: np.ndarray, labels: list[int]) -> None:
    from pytorch_grad_cam.utils.image import show_cam_on_image

    n = len(DEMO_FILES)
    col_titles = ["Original", "Grad-CAM", "ICD (hide salient)", "AICD (hide background)"]
    row_labels = DEMO_LABELS

    fig, axes = plt.subplots(n, 4, figsize=(11, 3.2 * n),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.04})

    for row in range(n):
        img_np = tensor_to_uint8(tensors[row])
        img_float = tensors[row].permute(1, 2, 0).numpy()
        smap = saliency[row]

        # Col 0: original
        axes[row, 0].imshow(img_np)

        # Col 1: GradCAM overlay
        overlay = show_cam_on_image(img_float, smap, use_rgb=True)
        axes[row, 1].imshow(overlay)

        # Col 2: ICD
        icd_np = apply_icd(model, img_np.copy(), labels[row])
        axes[row, 2].imshow(icd_np)

        # Col 3: AICD
        aicd_np = apply_aicd(model, img_np.copy(), labels[row])
        axes[row, 3].imshow(aicd_np)

        axes[row, 0].set_ylabel(row_labels[row], fontsize=9, labelpad=4)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, pad=6)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(OUT / "icd_aicd_comparison.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT / "icd_aicd_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved icd_aicd_comparison.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 2: fill strategies (Labrador only, ICD row + AICD row)
# ---------------------------------------------------------------------------

def figure_fill_strategies(model: NormalizedModel, tensors: torch.Tensor,
                           labels: list[int]) -> None:
    img_np = tensor_to_uint8(tensors[0])          # Labrador
    label = labels[0]
    fills = ["gaussian_blur", "local_mean", "noise", "solid"]
    fill_labels = ["Gaussian blur", "Local mean", "Gaussian noise", "Solid (zero)"]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.04})

    row_labels = ["ICD", "AICD"]

    # Column 0: original
    for row in range(2):
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(row_labels[row], fontsize=11, labelpad=4, fontweight="bold")
    axes[0, 0].set_title("Original", fontsize=10, pad=6)

    for col, (fill, flabel) in enumerate(zip(fills, fill_labels), start=1):
        icd_np = apply_icd(model, img_np.copy(), label, fill=fill)
        aicd_np = apply_aicd(model, img_np.copy(), label, fill=fill)
        axes[0, col].imshow(icd_np)
        axes[1, col].imshow(aicd_np)
        axes[0, col].set_title(flabel, fontsize=10, pad=6)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(OUT / "fill_strategies.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT / "fill_strategies.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved fill_strategies.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 3: tile size effect (Labrador, ICD only)
# ---------------------------------------------------------------------------

def figure_tile_sizes(model: NormalizedModel, tensors: torch.Tensor,
                      labels: list[int]) -> None:
    img_np = tensor_to_uint8(tensors[0])
    label = labels[0]
    tile_sizes = [4, 8, 16, 32]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2),
                             gridspec_kw={"wspace": 0.04})

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=10, pad=6)

    for col, ts in enumerate(tile_sizes, start=1):
        aug = apply_icd(model, img_np.copy(), label, tile_size=ts)
        axes[col].imshow(aug)
        axes[col].set_title(f"$t={ts}$", fontsize=10, pad=6)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(OUT / "tile_sizes.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT / "tile_sizes.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved tile_sizes.pdf / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model (ResNet-18, ImageNet weights)...")
    model = build_model()

    print("Loading demo images...")
    pil_imgs, tensors = load_images()

    print("Running inference to get predicted labels...")
    with torch.no_grad():
        logits = model(tensors)
    labels = logits.argmax(dim=1).tolist()
    print(f"Predicted label indices: {labels}")

    print("Computing Grad-CAM saliency maps...")
    saliency = compute_gradcam(model, tensors, labels)

    print("Generating Figure 1: ICD/AICD comparison grid...")
    figure_comparison(model, tensors, saliency, labels)

    print("Generating Figure 2: Fill strategies...")
    figure_fill_strategies(model, tensors, labels)

    print("Generating Figure 3: Tile size effect...")
    figure_tile_sizes(model, tensors, labels)

    print(f"\nDone. Figures saved to {OUT}")
