"""
RUN THIS ON YOUR LAPTOP, NOT IN THE SANDBOX -- it downloads
sentence-transformers/all-MiniLM-L6-v2 from Hugging Face on first run.

Embeds every child chunk in chunks.jsonl and saves vectors + a parallel
metadata list to disk as .npy / .json, so retrieval.py can load them without
re-embedding every time.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_NAME = "all-MiniLM-L6-v2"  # local, self-hosted, free at inference


def load_children():
    children = []
    with open(os.path.join(DATA_DIR, "chunks.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            if not c["is_parent"]:
                children.append(c)
    return children


def build_embeddings():
    print(f"Loading {MODEL_NAME} (first run downloads it from Hugging Face)...")
    model = SentenceTransformer(MODEL_NAME)

    children = load_children()
    texts = [c["text"] for c in children]

    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=False)

    np.save(os.path.join(DATA_DIR, "chunk_vectors.npy"), vectors)
    with open(os.path.join(DATA_DIR, "chunk_vector_index.json"), "w") as f:
        json.dump([c["chunk_id"] for c in children], f)

    print(f"Saved {vectors.shape} vectors to data/chunk_vectors.npy")


def embed_query(model, text: str) -> np.ndarray:
    return model.encode([text], normalize_embeddings=False)[0]


if __name__ == "__main__":
    build_embeddings()
