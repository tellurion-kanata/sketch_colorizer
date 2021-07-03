import torch
from PIL import Image

def save_image(data, filename):
    """
        image should be a torch.Tensor().cpu() [c, h, w]
        rgb value: [-1, 1] -> [0, 255]
    """

    img = (data.clone() + 1.) * 127.5

    img = img.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)
