# VisionQA AI Assistant

A multimodal AI application built with **FastAPI**, **Hugging Face Transformers**, **BLIP**, **CLIP**, and **FAISS**.

This project demonstrates three core computer vision capabilities:

* 🖼️ Image Captioning
* ❓ Visual Question Answering (VQA)
* 🔍 Semantic Image Search using CLIP embeddings and FAISS

---

## Features

* Upload an image and generate a natural language caption.
* Ask questions about an uploaded image.
* Search for visually similar images using vector embeddings.
* Automatic image embedding generation.
* Persistent FAISS vector index.
* Interactive API documentation with Swagger UI.

---

## Tech Stack

### Backend

* Python
* FastAPI
* PyTorch
* Hugging Face Transformers
* Pillow
* FAISS
* NumPy

### AI Models

* BLIP Image Captioning
* BLIP Visual Question Answering
* OpenAI CLIP (ViT-B/32)

---

## Project Structure

```text
vision-ai/

app/
│
├── main.py
├── inference.py
├── embeddings.py
├── vector_store.py
├── search.py
├── config.py
├── models.py
└── utils.py

uploads/
faiss/

requirements.txt
README.md
```

---

## Installation

### Clone the repository

```bash
git clone <repository-url>

cd vision-ai
```

---

### Create a virtual environment

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Mac/Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

If your project does not have an `app` folder and `main.py` is in the project root, run:

```bash
uvicorn main:app --reload
```

---

## Open the API

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

## API Endpoints

### Generate Caption

```
POST /caption
```

Input

* image

Response

```json
{
    "filename": "dog.jpg",
    "caption": "A brown dog running through a grassy field."
}
```

---

### Visual Question Answering

```
POST /ask
```

Input

* image
* question

Example question

```
What color is the car?
```

Response

```json
{
    "filename": "street.jpg",
    "question": "What color is the car?",
    "answer": "Blue"
}
```

---

### Semantic Image Search

```
POST /search
```

Input

* image

Response

```json
{
    "matches": [
        {
            "filename": "dog2.jpg",
            "similarity": 0.98
        },
        {
            "filename": "husky.jpg",
            "similarity": 0.95
        }
    ]
}
```

---

## How It Works

### Image Captioning

```
Image

↓

BLIP

↓

Caption
```

---

### Visual Question Answering

```
Image

+

Question

↓

BLIP VQA

↓

Answer
```

---

### Semantic Search

```
Image

↓

CLIP

↓

Embedding

↓

FAISS

↓

Nearest Neighbours
```

---



