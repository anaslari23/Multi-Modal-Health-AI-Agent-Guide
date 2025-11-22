#!/usr/bin/env python3
"""
Prepare RAG corpus from healthcare dataset.

This script extracts medical knowledge from the healthcare CSV dataset
and creates a corpus for the RAG (Retrieval-Augmented Generation) system.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def load_healthcare_data(csv_path: str) -> pd.DataFrame:
    """Load and clean healthcare dataset."""
    print(f"Loading healthcare data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Drop rows with missing critical fields
    df = df.dropna(subset=['Medical Condition', 'Medication'])
    print(f"After cleaning: {len(df)} records")
    
    return df


def create_medical_entries(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Create medical knowledge entries from patient records."""
    entries = []
    
    # Group by medical condition to create comprehensive entries
    for condition in df['Medical Condition'].unique():
        condition_df = df[df['Medical Condition'] == condition]
        
        # Get common medications for this condition
        medications = condition_df['Medication'].value_counts().head(5).index.tolist()
        
        # Get age statistics
        avg_age = condition_df['Age'].mean()
        
        # Get common test results
        test_results = condition_df['Test Results'].value_counts().to_dict()
        
        # Create entry
        entry = {
            "question": f"What are the symptoms and treatment for {condition}?",
            "answer": f"{condition} is a medical condition that affects patients with an average age of {avg_age:.1f} years. "
                     f"Common medications used for treatment include: {', '.join(medications)}. "
                     f"Test results typically show: {', '.join([f'{k} ({v} cases)' for k, v in list(test_results.items())[:3]])}.",
            "metadata": {
                "condition": condition,
                "sample_size": len(condition_df),
                "source": "healthcare_dataset"
            }
        }
        entries.append(entry)
        
        # Create medication-specific entries
        for med in medications[:3]:  # Top 3 medications
            med_entry = {
                "question": f"What medication is used for {condition}?",
                "answer": f"{med} is commonly prescribed for {condition}. "
                         f"It has been used in {len(condition_df[condition_df['Medication'] == med])} cases in our dataset.",
                "metadata": {
                    "condition": condition,
                    "medication": med,
                    "source": "healthcare_dataset"
                }
            }
            entries.append(med_entry)
    
    print(f"Created {len(entries)} medical knowledge entries")
    return entries


def append_to_corpus(entries: List[Dict[str, str]], corpus_path: Path):
    """Append entries to the RAG corpus file."""
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count existing entries
    existing_count = 0
    if corpus_path.exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
    
    print(f"Appending {len(entries)} entries to {corpus_path}")
    print(f"Existing entries: {existing_count}")
    
    with open(corpus_path, 'a', encoding='utf-8') as f:
        for entry in entries:
            # Format as JSONL
            json_line = json.dumps({
                "text": f"Q: {entry['question']}\nA: {entry['answer']}",
                "metadata": entry['metadata']
            })
            f.write(json_line + '\n')
    
    print(f"Total entries after append: {existing_count + len(entries)}")


def main():
    """Main execution function."""
    # Paths
    csv_path = "/Users/anaslari/Desktop/doctor_online/datasets/healthcare_dataset.csv"
    corpus_path = Path("/Users/anaslari/Desktop/doctor_online/mm-hie-backend/app/rag/datasets/corpus/merged_corpus.jsonl")
    
    # Load data
    df = load_healthcare_data(csv_path)
    
    # Create entries
    entries = create_medical_entries(df)
    
    # Append to corpus
    append_to_corpus(entries, corpus_path)
    
    print("\n✅ RAG corpus preparation complete!")
    print(f"📁 Corpus location: {corpus_path}")
    print("\nNext steps:")
    print("1. Rebuild FAISS index: python scripts/build_faiss_index.py")
    print("2. Test retrieval: python -c \"from app.rag.rag_engine import rag; print(rag.query('What medication is used for diabetes?'))\"")


if __name__ == "__main__":
    main()
