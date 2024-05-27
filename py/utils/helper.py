#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import torch
import numpy as np
from PIL import Image

class MS_Image:

    @staticmethod
    def empty(width, height):
        return torch.zeros((height, width, 3), dtype=torch.uint8),

    @staticmethod
    def tensor2pil(t_image: torch.Tensor)  -> Image:
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def pil2tensor(image:Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class MS_Mask:

    @staticmethod
    def empty(width, height):
        return torch.zeros((height, width), dtype=torch.uint8),

    # CREDITS ComfyUI-KJNodes\nodes\mask_nodes.py l.1201
    def resize(mask, width, height, keep_proportions):
        if keep_proportions:
            _, oh, ow = mask.shape
            width = ow if width == 0 else width
            height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow*ratio)
            height = round(oh*ratio)
        outputs = mask.unsqueeze(1)
        outputs = F.interpolate(outputs, size=(height, width), mode="nearest")
        outputs = outputs.squeeze(1)

        return(outputs, outputs.shape[2], outputs.shape[1],)