import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import FusionDataset
from .model import CrossModalTransformer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[truthy-function]
        return torch.device("mps")
    return torch.device("cpu")


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fusion/train.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--outdir", default="checkpoints/fusion")
    args = parser.parse_args()

    device = get_device()

    ds = FusionDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4)

    # infer num_labels from first record's labels length
    first_rec = ds.records[0]
    num_labels = int(len(first_rec["labels"]))

    model = CrossModalTransformer(num_labels=num_labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    os.makedirs(args.outdir, exist_ok=True)

    best_loss = float("inf")
    best_state = None

    for ep in range(args.epochs):
        model.train()
        lsum = 0.0

        all_logits = []
        all_labels = []
        all_risk_true = []
        all_risk_pred = []

        for x, labels, risk in loader:
            x = {k: v.to(device) for k, v in x.items()}
            labels = labels.to(device)
            risk = risk.to(device)

            logits, risk_pred, _ = model(x["nlp"], x["img"], x["labs"], x["vitals"])
            loss = bce(logits, labels) + 0.5 * mse(risk_pred, risk)

            opt.zero_grad()
            loss.backward()
            opt.step()

            lsum += float(loss.item())

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_risk_true.append(risk.detach().cpu())
            all_risk_pred.append(risk_pred.detach().cpu())

        epoch_loss = lsum / max(1, len(loader))

        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        risk_true_cat = torch.cat(all_risk_true, dim=0)
        risk_pred_cat = torch.cat(all_risk_pred, dim=0)

        probs = torch.sigmoid(logits_cat)
        preds = (probs > 0.5).float()

        # Top-1 / Top-3 accuracy based on predicted probabilities
        top1_preds = torch.topk(probs, k=1, dim=1).indices
        top3_preds = torch.topk(probs, k=min(3, probs.size(1)), dim=1).indices
        true_indices = labels_cat.nonzero(as_tuple=False)

        def topk_acc(k_indices: torch.Tensor) -> float:
            if true_indices.numel() == 0:
                return 0.0
            correct = 0
            for i in range(labels_cat.size(0)):
                true_lbls = (labels_cat[i] > 0.5).nonzero(as_tuple=False).view(-1)
                if true_lbls.numel() == 0:
                    continue
                if any(t.item() in k_indices[i].tolist() for t in true_lbls):
                    correct += 1
            return correct / max(1, labels_cat.size(0))

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

        print(
            f"Epoch {ep} loss: {epoch_loss:.4f} | top1: {top1:.3f} | top3: {top3:.3f} | "
            f"micro-F1: {f1:.3f} | Brier: {brier:.4f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()

    # Save best model
    ckpt_path = os.path.join(args.outdir, "best.pth")
    if best_state is not None:
        torch.save(best_state, ckpt_path)
    else:
        torch.save(model.state_dict(), ckpt_path)
    print(f"Saved fusion model checkpoint to {ckpt_path}")

    # Save fusion vectors for evaluation
    model.eval()
    all_fused = []
    all_labels_eval = []
    all_risk_eval = []
    with torch.no_grad():
        for x, labels, risk in loader:
            x = {k: v.to(device) for k, v in x.items()}
            logits, risk_pred, fused = model(x["nlp"], x["img"], x["labs"], x["vitals"])
            all_fused.append(fused.cpu().numpy())
            all_labels_eval.append(labels.numpy())
            all_risk_eval.append(risk.numpy())

    fused_arr = np.concatenate(all_fused, axis=0)
    labels_arr = np.concatenate(all_labels_eval, axis=0)
    risk_arr = np.concatenate(all_risk_eval, axis=0)
    out_path = os.path.join(args.outdir, "fusion_vectors.npz")
    np.savez(out_path, fused=fused_arr, labels=labels_arr, risk=risk_arr)
    print(f"Saved fusion vectors and labels to {out_path}")


if __name__ == "__main__":
    train()
