from fastapi import FastAPI, UploadFile, File
from PIL import Image

app = FastAPI()

@app.get("/")
def home():
    return{
        "message": "Vision AI API is running"
    }

# sending an image
@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    
    # Convert the uploaded file into a Pillow image
    pil_image = Image.open(image.file)
    
    # Return some information about the image
    return {
        "filename": image.filename,
        "width": pil_image.width,
        "height": pil_image.height,
        "mode": pil_image.mode,
        "format": pil_image.format
    }
