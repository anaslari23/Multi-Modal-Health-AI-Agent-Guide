import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH = str(BASE_DIR / "app/rag/datasets/training/medical_instruct.json")
OUTPUT_DIR = str(BASE_DIR / "models/lora-medical-adapter")

def train():
    print(f"Loading base model: {MODEL_NAME}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 4-bit to save memory (QLoRA)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Enable gradient checkpointing and prepare for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] # Target attention layers
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load Data
    print(f"Loading training data from {DATA_PATH}...")
    data_files = {"train": DATA_PATH}
    dataset = load_dataset("json", data_files=data_files, split="train")
    
    # Formatting function
    def format_prompt(sample):
        # Llama 3 Instruct format
        instruction = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sample['instruction']}<|eot_id|>"
        user = f"<|start_header_id|>user<|end_header_id|>\n\n{sample['input']}<|eot_id|>"
        assistant = f"<|start_header_id|>assistant<|end_header_id|>\n\n{sample['output']}<|eot_id|>"
        return {"text": instruction + user + assistant}

    dataset = dataset.map(format_prompt)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # Low batch size for local training
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100, # Short run for demo
        save_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_datasets,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    train()
