from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

import shutil

from app.config import UPLOAD_FOLDER
from app.inference import generate_caption, answer_question
from app.embeddings import get_image_embedding
from app.vector_store import add_image
from app.search import search_similar_images

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Vision AI API is running"
    }


# -------------------------------------
# Generate Caption
# -------------------------------------
@app.post("/caption")
async def caption_image(image: UploadFile = File(...)):
    try:

        save_path = UPLOAD_FOLDER / image.filename

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        pil_image = Image.open(save_path)

        caption = generate_caption(pil_image)

        embedding = get_image_embedding(pil_image)

        add_image(
            image.filename,
            embedding
        )

        return {
            "filename": image.filename,
            "caption": caption
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


# -------------------------------------
# Visual Question Answering
# -------------------------------------
@app.post("/ask")
async def ask_question(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    try:

        pil_image = Image.open(image.file)

        answer = answer_question(
            pil_image,
            question
        )

        return {
            "filename": image.filename,
            "question": question,
            "answer": answer
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e)
            }
        )


# -------------------------------------
# Search Similar Images
# -------------------------------------
@app.post("/search")
async def search_image(image: UploadFile = File(...)):
    try:

        pil_image = Image.open(image.file)

        embedding = get_image_embedding(pil_image)

        matches = search_similar_images(
            embedding,
            top_k=5
        )

        return {
            "matches": matches
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e)
            }
        )