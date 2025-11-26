import sys
import os
import time
import statistics
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "mm-hie-backend"))

from app.rag.rag_engine import rag

def benchmark():
    print("Initializing RAG engine...")
    engine = rag()
    
    # Warmup
    print("Warming up...")
    engine.query(question="What is the treatment for diabetes?", top_k=3)
    
    queries = [
        "What are the symptoms of hypertension?",
        "Explain the side effects of Metformin.",
        "How to diagnose pneumonia?",
        "Treatment for migraine?",
        "Contraindications for Aspirin?"
    ]
    
    latencies = []
    print(f"Running benchmark with {len(queries)} queries...")
    
    for i, q in enumerate(queries):
        start = time.time()
        engine.query(question=q, top_k=3)
        end = time.time()
        duration = end - start
        latencies.append(duration)
        print(f"Query {i+1}: {duration:.2f}s")
        
    avg_latency = statistics.mean(latencies)
    print(f"\nAverage Latency: {avg_latency:.2f}s")
    print(f"Min: {min(latencies):.2f}s, Max: {max(latencies):.2f}s")

if __name__ == "__main__":
    benchmark()
