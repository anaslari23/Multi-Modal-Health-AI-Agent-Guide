import json
import glob
from pathlib import Path
import pdfplumber

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data/raw_papers"
CORPUS_PATH = BASE_DIR / "app/rag/datasets/corpus/merged_corpus.jsonl"

def ingest_pdfs():
    """Reads all PDFs in data/raw_papers and appends to corpus."""
    pdf_files = glob.glob(str(PDF_DIR / "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDFs. Processing...")
    
    new_entries = []
    
    for pdf_path in pdf_files:
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if len(text) < 100:
                print(f"Skipping {pdf_path}: Text too short (scanned image?)")
                continue
                
            entry = {
                "title": Path(pdf_path).name,
                "source": "Medical Research Paper (PDF)",
                "content": text.strip()
            }
            new_entries.append(entry)
            print(f"Processed {pdf_path}")
            
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")

    if new_entries:
        print(f"Appending {len(new_entries)} entries to {CORPUS_PATH}...")
        with open(CORPUS_PATH, "a", encoding="utf-8") as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + "\n")
        print("Done.")
    else:
        print("No valid text extracted from PDFs.")

if __name__ == "__main__":
    ingest_pdfs()
