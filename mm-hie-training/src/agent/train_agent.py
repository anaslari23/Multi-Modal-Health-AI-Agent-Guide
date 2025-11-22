import os
os.environ["WANDB_DISABLED"] = "true"

import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

MODEL_NAME = os.environ.get("AGENT_MODEL", "t5-small")
DEFAULT_OUTDIR = os.environ.get("AGENT_OUTDIR", "checkpoints/agent")


def preprocess(examples, tokenizer, max_input: int = 512, max_output: int = 128):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)
    with tokenizer.as_target_tokenizer():  # type: ignore[attr-defined]
        labels = tokenizer(targets, max_length=max_output, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/agent/train.jsonl")
    parser.add_argument("--val_file", default="data/agent/val.jsonl")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    tokenized = ds.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=["input", "output"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_dir="logs/agent",
        learning_rate=args.lr,
        report_to=[],  # disable external loggers (incl. wandb)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.outdir)


if __name__ == "__main__":
    main()
