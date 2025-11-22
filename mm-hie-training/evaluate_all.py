import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent


def gather_status():
    checkpoints = {
        "nlp": BASE_DIR / "checkpoints" / "nlp",
        "imaging": BASE_DIR / "checkpoints" / "imaging",
        "vitals": BASE_DIR / "checkpoints" / "vitals",
        "fusion": BASE_DIR / "checkpoints" / "fusion",
        "agent": BASE_DIR / "checkpoints" / "agent",
    }

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "modalities": {},
    }

    for name, path in checkpoints.items():
        exists = path.exists()
        report["modalities"][name] = {
            "checkpoint_dir": str(path),
            "checkpoint_exists": bool(exists),
        }

    return report


def save_json(report):
    out_path = BASE_DIR / "evaluation_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote evaluation summary to {out_path}")


def save_plot(report):
    names = []
    values = []
    for name, info in report["modalities"].items():
        names.append(name)
        values.append(1.0 if info["checkpoint_exists"] else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("checkpoint_exists (1=yes, 0=no)")
    ax.set_title("Training status by modality")

    out_path = BASE_DIR / "evaluation_report.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote evaluation plot to {out_path}")


def main() -> None:
    report = gather_status()
    save_json(report)
    save_plot(report)


if __name__ == "__main__":
    main()
