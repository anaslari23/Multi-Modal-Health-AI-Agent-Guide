from datasets import load_dataset
import json, os


def format_sample(item):
    """Format a MedMCQA sample into instruction/output.

    The openlifescienceai MedMCQA dataset typically exposes `opa`..`opd` and
    `cop` (correct option text).
    """

    keys = ["opa", "opb", "opc", "opd"]
    if not all(k in item for k in keys):
        return None

    opts = {
        "A": item["opa"],
        "B": item["opb"],
        "C": item["opc"],
        "D": item["opd"],
    }

    correct = item.get("cop")
    if correct is None:
        return None
    correct = str(correct)

    prompt = (
        f"Question: {item['question']}\n"
        f"A. {opts['A']}\n"
        f"B. {opts['B']}\n"
        f"C. {opts['C']}\n"
        f"D. {opts['D']}\n\n"
        "Answer:"
    )

    return {
        "instruction": prompt,
        "output": correct,
    }


def main(outpath="processed/medmcqa.jsonl"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ds = load_dataset("openlifescienceai/medmcqa") if "openlifescienceai/medmcqa" in "openlifescienceai/medmcqa" else load_dataset("medmcqa")

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
            print(f"[preprocess_medmcqa] Skipped {skipped} samples due to unknown schema", flush=True)


if __name__ == "__main__":
    main()
