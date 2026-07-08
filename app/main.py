from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from inference import generate_caption

app = FastAPI()

@app.get("/")
def home():
    return{
        "message": "Vision AI API is running"
    }

# sending an image
@app.post("/caption")
async def caption_image(image: UploadFile = File(...)):
    
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
        
@app.post("/ask")
async def ask_question(
    image: UploadFile = File(...),
    # uses form for the next field since different inputs are being used
    question: str = Form(...)
):
    try:
        pil_image = Image.open(image.file)
        caption = generate_caption(pil_image)
        
        return {
            "file": image.filename,
            "caption": caption
        }
        
    except Exception as e:
        return JSONResponse(
            status_code = 400,
            content = {
                "error":str(e)
            }
        )
        
        
        
        
        
