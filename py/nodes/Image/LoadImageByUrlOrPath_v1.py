from PIL import Image, ImageSequence, ImageOps
import torch
import requests
from io import BytesIO
import os
import numpy as np
from inspect import cleandoc

def pil2tensor(img):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


def load_image(image_source):
    if image_source.startswith('http'):
        img = Image.open(requests.get(image_source, stream=True).raw)
        img = ImageOps.exif_transpose(img)
        # response = requests.get(image_source, timeout=10)
        # img = Image.open(BytesIO(response.content))
        file_name = image_source.split('/')[-1]
    else:
        img = Image.open(image_source)
        file_name = os.path.basename(image_source)
    return img, file_name


class LoadImageByUrlOrPath_v1:

    """
    # LoadImageByUrlOrPath_v1 : Load Image By Url Or Path

    Node to load image recursively from input directory

    ![LoadImage_v1](./MaraScottLoadImage_v1.png)

    ## Parameters

    - **image**: Input image to process

    """

    NAME = "Load Image By Url or Path"
    SHORTCUT = "i"

    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Example.jpg/800px-Example.jpg"})
            }
        }


    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "fn"
    CATEGORY = "image"

    def fn(self, url_or_path):
        print(url_or_path)
        img, name = load_image(url_or_path)
        img_out, mask_out = pil2tensor(img)
        return (img_out, mask_out)

