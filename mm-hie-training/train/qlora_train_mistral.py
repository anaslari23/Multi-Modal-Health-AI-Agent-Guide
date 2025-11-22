import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


def load_jsonl(path):
    return load_dataset("json", data_files=path)["train"]


def format_for_training(example):
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
    }


def main():
    cfg = yaml.safe_load(open("configs/qlora_mistral_medqa.yaml"))
    model_name = cfg["model_name"]

    # Some Mistral tokenizer configurations can trigger a tokenizers JSON
    # deserialization error with the fast tokenizer on certain platforms.
    # Fall back to the slow tokenizer to improve robustness.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_jsonl("processed/merged_medical.jsonl")
    ds = ds.map(format_for_training)

    use_cuda = torch.cuda.is_available()
    use_4bit = use_cuda

    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
        )

        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    bf16 = cfg["bf16"] and use_cuda
    fp16 = cfg.get("fp16", False) and use_cuda

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_steps=cfg["warmup_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=bf16,
        fp16=fp16,
        evaluation_strategy=cfg["evaluation_strategy"],
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
