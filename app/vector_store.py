import pickle

import faiss
import numpy as np

from app.config import (
    EMBEDDING_DIM,
    INDEX_FILE,
    METADATA_FILE
)

if INDEX_FILE.exists():
    index = faiss.read_index(str(INDEX_FILE))
else:
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    

if METADATA_FILE.exists():
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

else:
    metadata = []

def save():
    # Save the FAISS index and metadata to disk
    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
        
def add_image(filename, embedding):
    # Add a new image embedding to FAISS
    vector = embedding.cpu().numpy().astype(np.float32)
    vector = np.expand_dims(vector, axis=0)
    index.add(vector)
    metadata.append(filename)
    save()

def search(embedding, top_k = 3):
    # Search for the closest image embedding
    vector = embedding.cpu().numpy().astype(np.float32)
    vector = np.epand_dims(vector, axis = 0)
    scores, indices = index.search(vector, top_k)
    results = []
    for score, idx in zip(score[0], indices[0]):
        if idx == -1:
            continue
        
        results.append(
            {
                "filename": metadata[idx],
                "similarity": float(score),
            }
        )
    
    return results