import os
import argparse

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import load_efficientnet_b0
from .gradcam import save_gradcam_overlays


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/imaging")
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--outdir", default="checkpoints/imaging")
parser.add_argument("--export_onnx", action="store_true")
parser.add_argument("--onnx_path", default="checkpoints/imaging/model.onnx")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Torchvision transforms (for ImageFolder). Albumentations is used inside Grad-CAM utilities
train_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def train() -> None:
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Dataset layout: data_dir/train/<label>/image.jpg and data_dir/val/<label>/image.jpg
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    num_labels = len(train_ds.classes)
    model = load_efficientnet_b0(num_labels=num_labels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc = 0.0

    # Cache a small batch of validation images for Grad-CAM visualization
    val_examples = next(iter(val_loader)) if len(val_loader) > 0 else None

    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            # Convert class indices to one-hot for BCE loss
            labels_one_hot = torch.zeros(labels.size(0), num_labels, device=device)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

            logits = model(imgs)
            loss = criterion(logits, labels_one_hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_logits:
            print("No validation data found.")
            continue

        y_true_idx = np.concatenate(all_labels)
        y_preds = np.concatenate(all_logits)

        # Convert index labels to one-hot for metrics
        y_true = np.zeros_like(y_preds)
        y_true[np.arange(y_true.shape[0]), y_true_idx] = 1.0

        # Per-class AUC and mAP
        aucs = []
        aps = []
        for i in range(y_true.shape[1]):
            try:
                if len(np.unique(y_true[:, i])) < 2:
                    continue
                auc = roc_auc_score(y_true[:, i], y_preds[:, i])
                ap = average_precision_score(y_true[:, i], y_preds[:, i])
            except Exception:
                continue
            aucs.append(auc)
            aps.append(ap)

        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        mean_ap = float(np.mean(aps)) if aps else 0.0

        print(f"Epoch {epoch} mean_auc: {mean_auc:.4f}  mAP: {mean_ap:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_path = os.path.join(args.outdir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

            # Save a few Grad-CAM overlays for inspection
            if val_examples is not None:
                imgs_ex, labels_ex = val_examples
                save_gradcam_overlays(
                    model,
                    imgs_ex[:4],
                    labels_ex[:4],
                    out_dir=os.path.join(args.outdir, "cases"),
                    device=device,
                )

    # Optional ONNX export (gracefully handle missing onnx dependency)
    if args.export_onnx:
        dummy = torch.randn(1, 3, 224, 224, device=device)
        onnx_path = args.onnx_path
        try:
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                input_names=["input"],
                output_names=["logits"],
                opset_version=12,
            )
            print(f"Exported ONNX model to {onnx_path}")
        except (ModuleNotFoundError, torch.onnx.OnnxExporterError) as e:  # type: ignore[attr-defined]
            print(f"ONNX export skipped: {e}")

    print("Training complete")


if __name__ == "__main__":
    train()
