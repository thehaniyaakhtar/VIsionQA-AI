from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# When the model opens for the first time, Blip ins installed, weights are loaded into memory

# Load the processor (prepare image for the model)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Load the pretrained BLIP model
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Put the model in evaluation mode since only predictions are made
model.eval()

def generate_caption(image):
    # Generate a caption for a Pillow image
    # Convert the image into tensors
    
    # BLIP cannot use Pillow image directly
    # processor transforms it into format BLIP expects
    inputs = processor(
        image = image,
        return_tensors = "pt"
    )
    
    # Disable gradient calculations
    with torch.no_grad():
        output = model.generate(**inputs)
        # the model receives the processed image and predicts a sequence of words
        # result is still in form of token IDs
        
        
    # Convert generated token IDs into readable text
    caption = processor.decode(
        output[0],
        skip_special_tokens = True
    )
    
    return caption