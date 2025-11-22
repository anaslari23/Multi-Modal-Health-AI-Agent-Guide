# MM-HIE Training

This directory contains **training code, configs, and experiments** for the MM-HIE multimodal agent.

- `configs/` – YAML configs for each modality and the fusion / agent models.
- `data/` – Scripts for synthetic data and preparation of real datasets.
- `src/` – Training code for NLP, imaging, vitals, fusion, and the agent.
- `checkpoints/` – Saved model checkpoints.
- `logs/` – Training logs (TensorBoard, wandb, etc.).
- `notebooks/` – Experiment notebooks.
- `tests/` – Basic smoke tests for training pipelines.

This scaffold is intentionally lightweight and uses **plain PyTorch + simple scripts** so you can iterate quickly.
