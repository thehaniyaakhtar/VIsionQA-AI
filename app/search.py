import os
import torch
from os import Image

from app.embeddings import get_image_embedding

UPLOAD_FOLDER = "uploads"

def search_similar_images(query_image, top_k=3):
    # finding the most similar image inside the uploads
    # generating one embedding for the uploaded image
    query_embedding = get_image_embedding(query_image)
    similarities = []
    # Loop through image database, comparing every stored image against the vector
    for filename in os.listdir(UPLOAD_FOLDER):
        
        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )
        
        try:
            image = Image.open(filepath)
            
            image_embedding = get_image_embedding(image)
            
            similarity = torch.cosine_similarity(
                query_embedding,
                image_embedding,
                dim = 0
            )
            
            similarities.append(
                (
                    filename,
                    similarity.item()
                )
            )
        
        except Exception:
            continue
        
    similarities.sort(
        key = lambda x: x[1],
        reverse  = True
    )
    
    return similarities[:top_k]