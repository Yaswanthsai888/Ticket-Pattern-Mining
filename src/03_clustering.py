import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import warnings
warnings.filterwarnings('ignore')

def run_clustering(input_file, output_file, catalog_file):
    
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    
    print("Preparing embeddings...")
    # Convert list of arrays back to 2D numpy array
    embeddings = np.vstack(df['Embedding'].values)
    
    print("Running HDBSCAN clustering (this may take a moment)...")
    # Using scikit-learn HDBSCAN
    # Tuned parameters for dataset size (~1.4k tickets)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
    cluster_labels = hdbscan_model.fit_predict(embeddings)
    
    df['Cluster_ID'] = cluster_labels
    
    # -1 is noise in HDBSCAN
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Discovered {num_clusters} clusters. Noise points: {(cluster_labels == -1).sum()}")
    
    # Calculate Cluster Cohesion (Average Cosine Similarity)
    print("Calculating cluster cohesion and keywords...")
    cluster_data = []
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
            
        # Get indices for this cluster
        indices = df[df['Cluster_ID'] == cluster_id].index
        cluster_embeddings = embeddings[indices]
        
        # Cohesion
        if len(cluster_embeddings) > 1:
            sim_matrix = cosine_similarity(cluster_embeddings)
            # average off-diagonal elements
            mask = np.ones(sim_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            cohesion = sim_matrix[mask].mean()
        else:
            cohesion = 1.0
            
        # Keywords via TF-IDF
        cluster_texts = df.loc[indices, 'Lemma_Text'].tolist()
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            # Sum tfidf scores across all documents in cluster
            sum_tfidf = tfidf_matrix.sum(axis=0).A1
            top_indices = sum_tfidf.argsort()[-5:][::-1]
            top_keywords = ", ".join([feature_names[i] for i in top_indices])
        except ValueError:
            # If text is empty or too small
            top_keywords = "N/A"
            
        # Get top 3 domains/modules represented
        top_domains = df.loc[indices, 'Domain'].value_counts().head(2).index.tolist()
        
        cluster_data.append({
            'Cluster_ID': cluster_id,
            'Size': len(indices),
            'Top_Keywords': top_keywords,
            'Primary_Domains': " | ".join(top_domains),
            'Cohesion_Score': round(cohesion, 3)
        })
        
    catalog_df = pd.DataFrame(cluster_data)
    if not catalog_df.empty:
        catalog_df = catalog_df.sort_values(by='Size', ascending=False)
        print(f"Saving cluster catalog preliminary data to {catalog_file}...")
        catalog_df.to_csv(catalog_file, index=False)
    
    print(f"Saving clustered data to {output_file}...")
    df.to_parquet(output_file, engine='pyarrow', index=False)
    print("Clustering complete!")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python 03_clustering.py <input_file> <output_file> <catalog_file>")
        sys.exit(1)
    run_clustering(sys.argv[1], sys.argv[2], sys.argv[3])
