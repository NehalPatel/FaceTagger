import os
from PIL import Image

def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def resize_image(image_path, max_width=800):
    img = Image.open(image_path)
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int((float(img.height) * float(ratio)))
        img = img.resize((max_width, new_height), Image.ANTIALIAS)
        img.save(image_path)
