import json
import argparse
from pathlib import Path
from datasets import load_dataset

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "app/rag/datasets/corpus/merged_corpus.jsonl"
TRAIN_DATA_PATH = BASE_DIR / "app/rag/datasets/training/medical_instruct.json"

def fetch_healthcare_magic(n_samples=1000):
    """Fetches the ChatDoctor/HealthCareMagic dataset."""
    print(f"Fetching {n_samples} samples from lavita/ChatDoctor-HealthCareMagic-100k...")
    try:
        # Streaming=True allows us to not download the whole TBs if it was huge
        dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train", streaming=True)
        
        new_entries = []
        
        for i, item in enumerate(dataset):
            if i >= n_samples:
                break
            
            # Format: "input" (question) -> "output" (answer)
            entry = {
                "title": f"HealthCareMagic Case #{i}",
                "source": "ChatDoctor-HealthCareMagic-100k",
                "content": f"Patient: {item['input']}\nDoctor: {item['output']}"
            }
            new_entries.append(entry)
            
        return new_entries
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return []

def save_to_corpus(entries):
    """Appends to the RAG corpus."""
    print(f"Appending {len(entries)} entries to {CORPUS_PATH}...")
    with open(CORPUS_PATH, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

def save_for_training(entries):
    """Saves as a JSON file specifically for LoRA fine-tuning."""
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving training data to {TRAIN_DATA_PATH}...")
    
    # Convert to instruction format for Llama
    training_data = []
    for entry in entries:
        content = entry["content"]
        if "Patient:" in content and "Doctor:" in content:
            parts = content.split("Doctor:", 1)
            user_input = parts[0].replace("Patient:", "").strip()
            assistant_output = parts[1].strip()
            
            training_data.append({
                "instruction": "You are an expert doctor. Diagnose the patient.",
                "input": user_input,
                "output": assistant_output
            })
    
    with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to fetch")
    args = parser.parse_args()
    
    data = fetch_healthcare_magic(args.samples)
    if data:
        save_to_corpus(data)
        save_for_training(data)
        print("Done! RAG corpus updated and training data prepared.")
    else:
        print("No data fetched.")
