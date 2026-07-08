from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)
import torch

# When the model opens for the first time, Blip ins installed, weights are loaded into memory

# Load the processor (prepare image for the model)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Load the pretrained BLIP model
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

# Put the model in evaluation mode since only predictions are made
caption_model.eval()
vqa_model.eval()

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
        output = caption_model.generate(**inputs)
        # the model receives the processed image and predicts a sequence of words
        # result is still in form of token IDs
        
    return processor.decode(
        output[0],
        skip_special_tokens=True
    )

# passing 2 pieces of information
def answer_question(image, question):
    
    inputs = processor(
        images = image,
        text = question,
        return_tensors = "pt"
    )
    
    with torch.no_grad():
        output = vqa_model.generate(**inputs)
        
    return processor.decode(
        output[0],
        skip_special_tokens = True
    )