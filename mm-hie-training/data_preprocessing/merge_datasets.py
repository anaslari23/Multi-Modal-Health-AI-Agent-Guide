import json


def merge_files(files, outpath="processed/merged_medical.jsonl"):
    with open(outpath, "w") as outfile:
        for fname in files:
            with open(fname) as f:
                for line in f:
                    outfile.write(line)


if __name__ == "__main__":
    merge_files([
        "processed/medqa.jsonl",
        "processed/medmcqa.jsonl"
    ])
