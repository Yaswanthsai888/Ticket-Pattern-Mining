import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import warnings

warnings.filterwarnings("ignore")


def get_cluster_params(n):
    if n < 300:
        return 5, 2
    elif n < 2000:
        return 12, 4
    elif n < 10000:
        return 30, 10
    else:
        return 50, 15


def run_clustering(input_file, output_file, catalog_file):
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)

    embeddings = np.vstack(df["Embedding"].values)
    n = len(df)

    min_cluster_size, min_samples = get_cluster_params(n)

    print(f"Dataset size: {n}")
    print(f"Using HDBSCAN params: size={min_cluster_size}, samples={min_samples}")

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = model.fit_predict(embeddings)
    df["Cluster_ID"] = labels

    real_clusters = sorted(c for c in set(labels) if c != -1)

    print(f"Clusters found: {len(real_clusters)}")
    print(f"Noise points: {(labels == -1).sum()}")

    if len(real_clusters) == 0:
        df["Cluster_ID"] = 0
        real_clusters = [0]

    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    rows = []

    for cid in real_clusters:
        cluster_df = df[df["Cluster_ID"] == cid]
        idx = cluster_df.index.tolist()

        cluster_embeddings = embeddings[idx]

        if len(cluster_embeddings) > 1:
            sim = cosine_similarity(cluster_embeddings)
            mask = np.ones(sim.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            cohesion = sim[mask].mean()
        else:
            cohesion = 1.0

        texts = cluster_df["Lemma_Text"].fillna("").tolist()

        try:
            x = tfidf.fit_transform(texts)
            names = tfidf.get_feature_names_out()
            vals = x.sum(axis=0).A1
            top = vals.argsort()[-5:][::-1]
            keywords = ", ".join(names[i] for i in top)
        except:
            keywords = "N/A"

        top_domains = cluster_df["Domain"].value_counts().head(2).index.tolist()

        rows.append({
            "Cluster_ID": cid,
            "Size": len(cluster_df),
            "Top_Keywords": keywords,
            "Primary_Domains": " | ".join(top_domains),
            "Cohesion_Score": round(float(cohesion), 3),
            "Business_Priority": round(len(cluster_df) * cohesion, 2)
        })

    catalog = pd.DataFrame(rows).sort_values(
        by=["Business_Priority", "Size"],
        ascending=False
    )

    catalog.to_csv(catalog_file, index=False)
    df.to_parquet(output_file, index=False)

    print("Clustering complete.")


if __name__ == "__main__":
    run_clustering(sys.argv[1], sys.argv[2], sys.argv[3])