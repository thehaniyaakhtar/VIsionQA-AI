from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load the pretrained CLIP model and processor
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()

def get_image_embedding(image: Image.Image):
    '''
    Generate a feature vector for a Pillow image
    '''
    inputs = processor(
        images = image,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # get image features is called because u want vectors
        
        # Normalize the vector
        feature = features/features.norm(dim=-1, keepdim=True)
        
        return features.squeeze()
     # function returns images embeddings