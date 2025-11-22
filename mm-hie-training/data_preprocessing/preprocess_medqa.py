from datasets import load_dataset
import json, os


def format_sample(item):
    """Format a MedQA sample into instruction/output.

    The actual schema of openlifescienceai/MedQA-USMLE-4-options-hf is:
      - sent1: question stem
      - ending0..ending3: answer options
      - label: index (0-3) of correct option
    """

    opts = [item["ending0"], item["ending1"], item["ending2"], item["ending3"]]
    idx = int(item["label"])
    if idx < 0 or idx >= len(opts):
        return None
    correct = str(opts[idx])

    prompt = (
        f"Question: {item['sent1']}\n"
        f"A. {opts[0]}\n"
        f"B. {opts[1]}\n"
        f"C. {opts[2]}\n"
        f"D. {opts[3]}\n\n"
        "Answer:"
    )

    return {
        "instruction": prompt,
        "output": correct,
    }


def main(outpath="processed/medqa.jsonl"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf")

    with open(outpath, "w") as f:
        skipped = 0
        for split in ["train"]:
            for item in ds[split]:
                ex = format_sample(item)
                if ex is None:
                    skipped += 1
                    continue
                f.write(json.dumps(ex) + "\n")
        if skipped:
            print(f"[preprocess_medqa] Skipped {skipped} samples due to unknown schema", flush=True)


if __name__ == "__main__":
    main()
