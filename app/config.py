from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = BASE_DIR / "uploads"

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

VQA_MODEL = "Salesforce/blip-vqa-base"

CLIP_MODEL = "openai/clip-vit-base-patch32"