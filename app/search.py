from app.vector_store import search


def search_similar_images(embedding, top_k=3):
    """
    Search the FAISS index.
    """

    return search(
        embedding,
        top_k=top_k
    )