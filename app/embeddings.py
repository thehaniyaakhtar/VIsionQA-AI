from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

from app.config import CLIP_MODEL

# Load CLIP once when the server starts
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
model = CLIPModel.from_pretrained(CLIP_MODEL)

model.eval()


def get_image_embedding(image: Image.Image):
    """
    Generate a normalized CLIP embedding.
    """

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    # Normalize for cosine similarity
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.squeeze()