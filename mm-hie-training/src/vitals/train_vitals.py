import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import VitalsWindowDataset
from .model import build_vitals_model, contrastive_loss_stub


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[truthy-function]
        return torch.device("mps")
    return torch.device("cpu")


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/vitals/train.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", default="checkpoints/vitals")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    args = parser.parse_args()

    device = get_device()

    ds = VitalsWindowDataset(args.data, window_size=args.window, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4)

    num_features = ds.x.shape[-1]
    model = build_vitals_model(num_features=num_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()

    os.makedirs(args.outdir, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            risk_logits, anomaly_logits, emb = model(x)

            # Use the same label as proxy for both vitals risk and anomaly tag for now
            risk_loss = bce_loss(risk_logits, y)
            anomaly_loss = bce_loss(anomaly_logits, y)

            if args.contrastive_weight > 0.0:
                c_loss = contrastive_loss_stub(emb)
            else:
                c_loss = emb.new_zeros(())

            loss = risk_loss + anomaly_loss + args.contrastive_weight * c_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

        print(f"Epoch {ep} loss: {epoch_loss / max(1, len(loader))}")

    # Save checkpoint
    ckpt_path = os.path.join(args.outdir, "best.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved vitals model checkpoint to {ckpt_path}")

    # Also save embeddings + scores for fusion
    model.eval()
    all_emb = []
    all_risk = []
    all_anom = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            risk_logits, anomaly_logits, emb = model(x)
            all_emb.append(emb.cpu().numpy())
            all_risk.append(torch.sigmoid(risk_logits).cpu().numpy())
            all_anom.append(torch.sigmoid(anomaly_logits).cpu().numpy())

    emb_arr = np.concatenate(all_emb, axis=0)
    risk_arr = np.concatenate(all_risk, axis=0)
    anom_arr = np.concatenate(all_anom, axis=0)
    emb_path = os.path.join(args.outdir, "vitals_embeddings.npz")
    np.savez(emb_path, emb=emb_arr, vitals_risk=risk_arr, anomaly=anom_arr)
    print(f"Saved vitals embeddings and scores to {emb_path}")


if __name__ == "__main__":
    train()
