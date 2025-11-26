import sys
import os
import time
import statistics
from pathlib import Path

# Ensure we can import app
# We assume we are running from mm-hie-backend directory
sys.path.append(os.getcwd())

from app.rag.rag_engine import rag

def benchmark():
    print("Initializing RAG engine...")
    try:
        engine = rag()
    except Exception as e:
        print(f"Failed to initialize RAG engine: {e}")
        return

    # Warmup
    print("Warming up...")
    try:
        engine.query(question="What is the treatment for diabetes?", top_k=3)
    except Exception as e:
        print(f"Warmup failed: {e}")
        # Continue anyway to see if it's a transient issue or config issue
        
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
        try:
            result = engine.query(question=q, top_k=3)
            end = time.time()
            duration = end - start
            latencies.append(duration)
            print(f"Query {i+1}: {duration:.2f}s")
            print(f"Answer: {result['answer'][:500]}...")
            print("-" * 40)
        except Exception as e:
            print(f"Query {i+1} failed: {e}")
        
    if latencies:
        avg_latency = statistics.mean(latencies)
        print(f"\nAverage Latency: {avg_latency:.2f}s")
        print(f"Min: {min(latencies):.2f}s, Max: {max(latencies):.2f}s")
    else:
        print("\nNo queries succeeded.")

if __name__ == "__main__":
    benchmark()
