import csv
import random
import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "data" / "labs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "labs.csv"

    headers = ["patient_id", "Hemoglobin", "WBC", "Platelets", "CRP"]
    rows = []

    for i in range(args.rows):
        hb = round(random.uniform(7.0, 18.0), 1)
        wbc = random.randint(3000, 25000)
        plate = random.randint(50_000, 450_000)
        crp = random.randint(0, 200)
        rows.append([f"P{i}", hb, wbc, plate, crp])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Wrote labs CSV with {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
