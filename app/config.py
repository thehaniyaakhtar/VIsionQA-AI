from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Upload folder
UPLOAD_FOLDER = BASE_DIR / "uploads"

# FAISS storage
FAISS_FOLDER = BASE_DIR / "faiss"

INDEX_FILE = FAISS_FOLDER / "image.index"
METADATA_FILE = FAISS_FOLDER / "metadata.pkl"

# Models
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
VQA_MODEL = "Salesforce/blip-vqa-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Embedding size for CLIP
EMBEDDING_DIM = 512
