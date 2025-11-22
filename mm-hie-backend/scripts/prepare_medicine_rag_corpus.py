import csv
import json
import sys
import os
from pathlib import Path

def prepare_medicine_rag_corpus():
    """
    Convert allopathy_medicines_plus_1000.csv into RAG-compatible JSONL format.
    Appends to existing merged_corpus.jsonl.
    """
    
    # Paths
    csv_path = Path("../datasets/medicine/allopathy_medicines_plus_1000.csv")
    corpus_path = Path("app/rag/datasets/corpus/merged_corpus.jsonl")
    
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    
    print(f"📖 Reading medicine data from {csv_path}...")
    
    medicine_entries = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Extract key fields
            medicine_name = row.get('medicine_name', '').strip()
            generic_name = row.get('generic_name', '').strip()
            disease_condition = row.get('disease_condition', '').strip()
            therapeutic_category = row.get('therapeutic_category', '').strip()
            dosage_form = row.get('dosage_form', '').strip()
            strength = row.get('strength', '').strip()
            route = row.get('route_of_administration', '').strip()
            side_effects = row.get('common_side_effects', '').strip()
            contraindications = row.get('contraindications', '').strip()
            interactions = row.get('drug_interactions', '').strip()
            usage_frequency = row.get('usage_frequency', '').strip()
            storage = row.get('storage_conditions', '').strip()
            
            # Skip if no medicine name or disease condition
            if not medicine_name and not generic_name:
                continue
            
            # Use generic name if medicine name is empty
            med_name = medicine_name if medicine_name else generic_name
            
            # Create title
            if disease_condition:
                title = f"{med_name} for {disease_condition}"
            else:
                title = f"{med_name} - {therapeutic_category}" if therapeutic_category else med_name
            
            # Build comprehensive content
            content_parts = []
            
            # Basic info
            if generic_name and medicine_name:
                content_parts.append(f"Medicine: {medicine_name} (Generic: {generic_name})")
            else:
                content_parts.append(f"Medicine: {med_name}")
            
            if therapeutic_category:
                content_parts.append(f"Therapeutic Category: {therapeutic_category}")
            
            if disease_condition:
                content_parts.append(f"Disease Conditions: {disease_condition}")
            
            # Dosage info
            dosage_info = []
            if dosage_form:
                dosage_info.append(f"Form: {dosage_form}")
            if strength:
                dosage_info.append(f"Strength: {strength}")
            if route:
                dosage_info.append(f"Route: {route}")
            if usage_frequency:
                dosage_info.append(f"Frequency: {usage_frequency}")
            
            if dosage_info:
                content_parts.append("Dosage: " + ", ".join(dosage_info))
            
            # Safety information (CRITICAL)
            if contraindications:
                content_parts.append(f"⚠️ CONTRAINDICATIONS: {contraindications}")
            
            if side_effects:
                content_parts.append(f"Common Side Effects: {side_effects}")
            
            if interactions:
                content_parts.append(f"Drug Interactions: {interactions}")
            
            if storage:
                content_parts.append(f"Storage: {storage}")
            
            # Add disclaimer
            content_parts.append("IMPORTANT: This information is for educational purposes only. Consult a qualified physician before taking any medication.")
            
            content = ". ".join(content_parts)
            
            # Create JSONL entry
            entry = {
                "title": title,
                "source": "Medicine Database - Allopathy",
                "content": content
            }
            
            medicine_entries.append(entry)
    
    print(f"✅ Processed {len(medicine_entries)} medicine entries")
    
    # Append to existing corpus
    print(f"📝 Appending to {corpus_path}...")
    
    with open(corpus_path, 'a', encoding='utf-8') as f:
        for entry in medicine_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"✅ Successfully added {len(medicine_entries)} medicine entries to RAG corpus!")
    print(f"📊 Total corpus size: {sum(1 for _ in open(corpus_path))} entries")
    
    return len(medicine_entries)

if __name__ == "__main__":
    try:
        count = prepare_medicine_rag_corpus()
        print(f"\n🎉 Medicine RAG corpus preparation complete!")
        print(f"📌 Next steps:")
        print(f"   1. Delete vector index: rm -rf app/rag/datasets/corpus/vector_index")
        print(f"   2. Restart backend server to rebuild index")
        print(f"   3. Test chatbot with medicine queries")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
