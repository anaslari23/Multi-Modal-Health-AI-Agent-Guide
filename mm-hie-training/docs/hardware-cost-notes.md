# Hardware & Cost Notes

## Training Time Estimates

### NLP Fine-tuning (ClinicalBERT)
- **A100 GPU**: 4–8 hours
- **V100 GPU**: 1–2 days
- **Factors**: Dataset size, sequence length, number of epochs

### Imaging Fine-tuning
- **Large datasets**: 12–48 hours
- **Small proof-of-concepts**: Few hours
- **Factors**: Image resolution, model size (ResNet vs EfficientNet), dataset size

### Fusion Training
- **Recommended**: Medium GPU (A100 32GB or 16GB)
- **Training time**: Varies based on embedding dimensions and fusion architecture

## Cloud Options

### AWS EC2
- **p3.2xlarge** (V100): ~$3.06/hour
- **p3.8xlarge** (4x V100): ~$12.24/hour
- **p4d.24xlarge** (8x A100): ~$32.77/hour

### GCP
- **A2** (A100): ~$3.67/hour (1 GPU) to ~$29.36/hour (8 GPUs)
- **Preemptible instances**: 60-80% discount for fault-tolerant training

### Google Colab
- **Colab Pro**: ~$10/month for A100/V100 access
- **Colab Pro+**: ~$50/month for better uptime and more resources
- **Good for**: Experimentation, small datasets, prototyping

## Cost Optimization Tips

1. **Use smaller models for prototyping**: BERT-base instead of large, ResNet-34 instead of 152
2. **Mixed precision training**: Reduces memory usage, speeds up training
3. **Gradient checkpointing**: Trades compute for memory
4. **Early stopping**: Monitor validation loss to stop training early
5. **Hyperparameter tuning**: Use smaller subsets for initial sweeps
6. **Spot/preemptible instances**: 60-80% cost reduction for fault-tolerant jobs

## Local Hardware Options

### Development Setup
- **RTX 3090/4090**: 24GB VRAM, good for medium-sized models
- **RTX 4080**: 16GB VRAM, cost-effective option
- **M1/M2 Mac**: Good for NLP (MPS backend), limited for large imaging models

### Production Training
- **Multi-GPU setup**: 2-4x RTX 4090s for cost-effective training
- **Workstation with A100**: Enterprise option for consistent performance

## Monitoring Costs

- Use cloud cost alerts and budgets
- Monitor GPU utilization (target >80% for cost efficiency)
- Track training progress to estimate completion time
- Consider using Weights & Biases or MLflow for experiment tracking
