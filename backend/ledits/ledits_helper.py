import os
import numpy as np
import requests
from PIL import Image

def load_image(url):
    """Download and load an image as 512x512 RGB."""
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image.resize((512, 512))

def image_grid(imgs, rows, cols, spacing=20):
    """Arrange multiple images into a grid."""
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGBA", size=(cols * w + (cols - 1) * spacing, rows * h + (rows - 1) * spacing), color=(255, 255, 255, 0))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i // rows * (w + spacing), i % rows * (h + spacing)))
    return grid