from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
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
    
    try:
    
        # Convert the uploaded file into a Pillow image
        pil_image = Image.open(image.file)
        
        # Ask BLIP to generate a caption
        caption = generate_caption(pil_image)
        
        # Return some information about the image
        return {
            "filename": image.filename,
            "caption": caption
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content = {
                "error": str(e)
            }
        )
