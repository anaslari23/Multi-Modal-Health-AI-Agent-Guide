from __future__ import annotations

"""Placeholder script for preparing clinical data (e.g., MIMIC-III/IV).

This is intentionally a stub; fill in paths and preprocessing as needed.
"""

from pathlib import Path


def main() -> None:
  data_root = Path("./mm-hie-training/data/raw_mimic")
  out_root = Path("./mm-hie-training/data/processed_mimic")
  out_root.mkdir(parents=True, exist_ok=True)

  print(f"This is a placeholder; implement MIMIC preparation from {data_root} to {out_root}.")


if __name__ == "__main__":
  main()
