from pathlib import Path
import shutil
from PIL import Image


def save_image(upload_file, upload_folder):
    """
    Save an uploaded image and return
    the Pillow image together with its path.
    """

    path = Path(upload_folder) / upload_file.filename

    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return Image.open(path), path