import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Legacy vs DBB Ticket Pattern Mining Pipeline")
    parser.add_argument("input_file", help="Path to the raw Excel/CSV dataset")
    parser.add_argument("--output_dir", default="data", help="Base directory for outputs")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        sys.exit(1)
        
    print(f"Starting pipeline for dataset: {input_file}")
    
    # Define paths
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    base_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(os.path.join(base_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "output"), exist_ok=True)
    
    classified_file = os.path.join(base_dir, "processed", "classified_tickets.xlsx")
    normalized_file = os.path.join(base_dir, "processed", "tickets.parquet")
    vectorized_file = os.path.join(base_dir, "processed", "tickets_vectorized.parquet")
    clustered_file = os.path.join(base_dir, "processed", "tickets_clustered.parquet")
    catalog_file = os.path.join(base_dir, "output", "cluster_catalog.csv")
    pivot_file = os.path.join(base_dir, "output", "legacy_vs_dbb_pivot.csv")
    summary_file = os.path.join(base_dir, "output", "executive_summary.json")
    
    python_exe = sys.executable
    
    steps = [
        ("00: Classification", ["src/00_classify_tickets.py", input_file, classified_file]),
        ("01: Ingestion & Normalization", ["src/01_ingest_normalize.py", classified_file, normalized_file]),
        ("02: Preprocessing & Vectorization", ["src/02_preprocess_vectorize.py", normalized_file, vectorized_file]),
        ("03: Clustering", ["src/03_clustering.py", vectorized_file, clustered_file, catalog_file]),
        ("04: Metrics & Export", ["src/04_metrics_export.py", clustered_file, catalog_file, pivot_file]),
        ("05: LLM Strategic Naming", ["src/05_llm_naming.py", catalog_file, clustered_file]),
        ("06: Executive Summary", ["src/06_executive_summary.py", catalog_file, clustered_file, summary_file])
    ]
    
    for step_name, cmd in steps:
        print(f"\n{'='*50}\nRunning Step {step_name}\n{'='*50}")
        result = subprocess.run([python_exe] + cmd)
        if result.returncode != 0:
            print(f"\nPipeline failed at {step_name} (Exit code: {result.returncode})")
            sys.exit(result.returncode)
            
    print("\n✅ Pipeline completed successfully!")
    print(f"Launch dashboard with: streamlit run src/dashboard.py")

if __name__ == "__main__":
    main()
