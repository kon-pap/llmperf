from PIL import Image, ImageFile

def load_image(path: str) -> ImageFile.ImageFile:
    return Image.open(path)