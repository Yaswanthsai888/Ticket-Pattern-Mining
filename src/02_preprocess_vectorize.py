import pandas as pd
import numpy as np
import spacy
import re
import os
from sentence_transformers import SentenceTransformer
import sys
import warnings
warnings.filterwarnings('ignore')

def preprocess_and_vectorize(input_file, output_file):
    
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    
    # 1. Text Preprocessing
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model en_core_web_sm...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    print("Preprocessing text...")
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'\[URL\]', ' ', text)
        # Extract and remove IP addresses and Phone numbers to de-noise
        text = re.sub(r'\[IP\]|\[PHONE\]|\[TICKET\]', ' ', text)
        text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', text)
        
        # Keep basic structure but lowercase
        text = text.lower()
        # Normalise whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_error_codes(text):
        if not isinstance(text, str):
            return ""
        # Look for ERR-123, Exception, Traceback, HTTP codes
        matches = re.findall(r'err[_-]?\d{2,6}|exception|traceback|error code \d+', text.lower())
        return " ".join(set(matches))

    # Combine text fields for full context embedding
    df['Full_Text'] = df['Short_Description'].fillna('') + " " + df['Description_Text'].fillna('') + " " + df['Resolution_Notes'].fillna('')
    df['Clean_Text'] = df['Full_Text'].apply(clean_text)
    
    print("Extracting structured tokens...")
    df['Extracted_Errors'] = df['Full_Text'].apply(extract_error_codes)
    
    # Optional: Lemmatization for TF-IDF keyword extraction later
    def lemmatize_text(text):
        # Cap length for speed to avoid massive logs blocking it
        doc = nlp(text[:5000]) 
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return " ".join(tokens)
        
    print("Lemmatizing for keyword extraction...")
    df['Lemma_Text'] = df['Clean_Text'].apply(lemmatize_text)

    # 2. Vectorization
    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    # Prefer the local cache first so the pipeline can run without network access.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)

    try:
        model = SentenceTransformer('all-mpnet-base-v2', local_files_only=True)
    except Exception:
        model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Encoding texts to vectors (this may take a minute)...")
    texts_to_encode = df['Clean_Text'].tolist()
    embeddings = model.encode(texts_to_encode, batch_size=32, show_progress_bar=True)
    
    # Store embeddings as a list in the dataframe
    df['Embedding'] = list(embeddings)
    
    print(f"Saving vectorized data to {output_file}...")
    df.to_parquet(output_file, engine='pyarrow', index=False)
    print("Preprocessing and Vectorization complete!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 02_preprocess_vectorize.py <input_file> <output_file>")
        sys.exit(1)
    preprocess_and_vectorize(sys.argv[1], sys.argv[2])
