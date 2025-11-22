from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FusionDataset
from .model import CrossModalTransformer
from src.utils.calibration import calibration_curve, temperature_scale


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[truthy-function]
        return torch.device("mps")
    return torch.device("cpu")


def evaluate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fusion/train.npz")
    parser.add_argument("--ckpt", default="checkpoints/fusion/best.pth")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--outdir", default="checkpoints/fusion")
    args = parser.parse_args()

    device = get_device()

    ds = FusionDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)

    first_rec = ds.records[0]
    num_labels = int(len(first_rec["labels"]))

    model = CrossModalTransformer(num_labels=num_labels).to(device)
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state)
    model.eval()

    all_logits = []
    all_labels = []
    all_risk_true = []
    all_risk_pred = []

    with torch.no_grad():
        for x, labels, risk in loader:
            x = {k: v.to(device) for k, v in x.items()}
            labels = labels.to(device)
            risk = risk.to(device)

            logits, risk_pred, _ = model(x["nlp"], x["img"], x["labs"], x["vitals"])

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_risk_true.append(risk.cpu())
            all_risk_pred.append(risk_pred.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    risk_true_cat = torch.cat(all_risk_true, dim=0)
    risk_pred_cat = torch.cat(all_risk_pred, dim=0)

    probs = torch.sigmoid(logits_cat)
    preds = (probs > 0.5).float()

    # Top-1 / Top-3 accuracy
    top1_preds = torch.topk(probs, k=1, dim=1).indices
    top3_preds = torch.topk(probs, k=min(3, probs.size(1)), dim=1).indices

    def topk_acc(k_indices: torch.Tensor) -> float:
        correct = 0
        total = labels_cat.size(0)
        for i in range(total):
            true_lbls = (labels_cat[i] > 0.5).nonzero(as_tuple=False).view(-1)
            if true_lbls.numel() == 0:
                continue
            if any(t.item() in k_indices[i].tolist() for t in true_lbls):
                correct += 1
        return correct / max(1, total)

    top1 = topk_acc(top1_preds)
    top3 = topk_acc(top3_preds)

    # Micro F1
    tp = (preds * labels_cat).sum().item()
    fp = (preds * (1 - labels_cat)).sum().item()
    fn = ((1 - preds) * labels_cat).sum().item()
    precision = tp / max(1e-8, tp + fp)
    recall = tp / max(1e-8, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    # Brier score for risk regression
    risk_true_np = risk_true_cat.numpy()
    risk_pred_np = risk_pred_cat.numpy()
    brier = float(np.mean((risk_pred_np - risk_true_np) ** 2))

    # Calibration for multi-label logits (temperature scaling + curve per first label)
    logits_np = logits_cat.numpy()
    labels_np = labels_cat.numpy()
    temp = temperature_scale(logits_np, labels_np)
    calibrated_logits = logits_np / temp
    calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))

    # Use the first label as a representative for reliability curve
    calib = calibration_curve(labels_np[:, 0], calibrated_probs[:, 0], n_bins=10)

    os.makedirs(args.outdir, exist_ok=True)
    metrics_path = os.path.join(args.outdir, "fusion_eval_metrics.npz")
    np.savez(
        metrics_path,
        top1=top1,
        top3=top3,
        micro_f1=f1,
        brier=brier,
        temperature=temp,
        calib_prob_true=np.array(calib["prob_true"]),
        calib_prob_pred=np.array(calib["prob_pred"]),
    )

    print(
        f"Fusion eval -> top1: {top1:.3f}, top3: {top3:.3f}, micro-F1: {f1:.3f}, "
        f"Brier: {brier:.4f}, temperature: {temp:.3f}"
    )
    print(f"Saved fusion evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    evaluate()
