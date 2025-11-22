# Fine-Tuning Guide: Medical LLM for Medicine Suggestions

## Overview

This guide explains how to use the provided Google Colab notebook to fine-tune a medical language model with your medicine dataset, making the chatbot significantly more accurate for medicine suggestions.

## What You'll Get

✅ **Highly accurate medicine suggestions** based on diseases  
✅ **Contraindication awareness** - knows when NOT to suggest medicines  
✅ **Side effects knowledge** - warns about common side effects  
✅ **Drug interaction awareness** - understands medicine interactions  
✅ **Small, efficient model** (~1-2GB) that runs on CPU/GPU  

---

## Quick Start

### 1. Open the Notebook in Google Colab

1. Upload `Medical_LLM_Fine_Tuning.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4)

### 2. Upload Your Dataset

- Run the upload cell
- Select `allopathy_medicines_plus_1000.csv`
- The notebook will generate **~3,000 training examples** from 1,027 medicines

### 3. Start Training

- Click "Runtime → Run all"
- Training takes **~15-20 minutes** on free Colab GPU
- The model will learn:
  - Disease → Medicine mappings
  - Contraindications for each medicine
  - Side effects and warnings
  - Drug interactions

### 4. Download the Model

- After training, the notebook automatically downloads `medical-medicine-lora-final.zip`
- This contains the fine-tuned model adapters

---

## Integration with Your Project

### Option 1: Use LoRA Adapters (Recommended)

1. **Extract the downloaded zip**:
   ```bash
   unzip medical-medicine-lora-final.zip
   ```

2. **Copy to your project**:
   ```bash
   cp -r medical-medicine-lora-final mm-hie-backend/models/
   ```

3. **Update RAG engine** (`mm-hie-backend/app/rag/rag_engine.py`):
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel
   
   # Load base model
   base_model = AutoModelForCausalLM.from_pretrained(
       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
       device_map="auto"
   )
   
   # Load fine-tuned adapters
   model = PeftModel.from_pretrained(
       base_model, 
       "./models/medical-medicine-lora-final"
   )
   
   tokenizer = AutoTokenizer.from_pretrained(
       "./models/medical-medicine-lora-final"
   )
   ```

### Option 2: Use Merged Model (Simpler)

The notebook also creates a merged model that's easier to use:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./models/medical-medicine-merged",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./models/medical-medicine-merged"
)
```

---

## Training Details

### Model Architecture

- **Base Model**: TinyLlama-1.1B-Chat (lightweight, fast)
- **Fine-Tuning Method**: QLoRA (4-bit quantization)
- **LoRA Rank**: 16
- **Trainable Parameters**: ~0.5% of total (very efficient!)

### Dataset Format

The notebook converts your CSV into instruction-following format:

```
### Instruction:
What medicine can be used for fever?

### Response:
Paracetamol is used for fever. It belongs to the Analgesic, Antipyretic category. 
⚠️ Contraindications: Severe hepatic impairment. 
Common side effects: Generally well tolerated, rare hepatotoxicity with overdose. 
Always consult a physician before taking any medication.
```

### Training Configuration

- **Epochs**: 3
- **Batch Size**: 4 (with gradient accumulation)
- **Learning Rate**: 2e-4
- **Optimizer**: Paged AdamW 8-bit
- **GPU Memory**: ~6-8GB (fits on free Colab T4)

---

## Testing the Model

The notebook includes test cells to verify the model works:

```python
# Test queries
queries = [
    "What medicine can be used for fever?",
    "Tell me about Paracetamol",
    "What are the contraindications for Ibuprofen?",
    "Suggest medicine for bacterial infection"
]

for query in queries:
    response = generate_response(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

---

## Advanced Options

### 1. Use a Larger Model

For better accuracy, replace TinyLlama with:

```python
# In the notebook, change model_name to:
model_name = "microsoft/BioGPT-Large"  # Medical domain model
# OR
model_name = "meta-llama/Llama-2-7b-hf"  # Requires HuggingFace token
```

### 2. Train Longer

```python
# Increase epochs for better learning
training_args = TrainingArguments(
    num_train_epochs=5,  # Instead of 3
    ...
)
```

### 3. Add More Data

- Combine with other medical datasets
- Add custom medicine-disease pairs
- Include patient case studies

---

## Expected Results

### Before Fine-Tuning (RAG only)
```
Q: What medicine for fever?
A: Common medicines for fever include antipyretics. Consult a doctor.
```

### After Fine-Tuning
```
Q: What medicine for fever?
A: Paracetamol is used for fever. It belongs to the Analgesic, Antipyretic category.
⚠️ Contraindications: Severe hepatic impairment.
Common side effects: Generally well tolerated, rare hepatotoxicity with overdose.
Always consult a physician before taking any medication.
```

---

## Troubleshooting

### Out of Memory Error

- Reduce batch size: `per_device_train_batch_size=2`
- Reduce sequence length: `max_seq_length=256`
- Use smaller LoRA rank: `r=8`

### Training Too Slow

- Use Colab Pro for better GPU (A100)
- Reduce dataset size for testing
- Decrease epochs to 2

### Model Not Learning

- Increase learning rate: `learning_rate=3e-4`
- Train for more epochs: `num_train_epochs=5`
- Check dataset quality

---

## Next Steps

1. ✅ Run the Colab notebook
2. ✅ Download the fine-tuned model
3. ✅ Integrate into your project
4. ✅ Test with real queries
5. ✅ Deploy and monitor performance

---

## Resources

- **Colab Notebook**: `Medical_LLM_Fine_Tuning.ipynb`
- **Dataset**: `allopathy_medicines_plus_1000.csv`
- **Documentation**: [Hugging Face PEFT](https://huggingface.co/docs/peft)
- **LoRA Paper**: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

---

## Support

If you encounter issues:
1. Check the notebook's test cells
2. Verify GPU is enabled in Colab
3. Ensure dataset uploaded correctly
4. Review training logs for errors

**Happy Fine-Tuning! 🚀**
