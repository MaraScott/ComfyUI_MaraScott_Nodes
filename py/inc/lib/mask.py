#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont

from ...utils.log import *


class MS_Mask:

    @staticmethod
    def empty(width, height):
        return (torch.zeros((height, width), dtype=torch.float32, device="cpu")).unsqueeze(0)

    # CREDITS ComfyUI-KJNodes\nodes\mask_nodes.py l.1201
    def resize(mask, width, height, keep_proportions):
        if keep_proportions:
            _, oh, ow = mask.shape
            width = ow if width == 0 else width
            height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow * ratio)
            height = round(oh * ratio)
        outputs = mask.unsqueeze(1)
        outputs = F.interpolate(outputs, size=(height, width), mode="nearest")
        outputs = outputs.squeeze(1)

        return outputs, outputs.shape[2], outputs.shape[1]
    
    # PIL to Mask
    def pil2mask(self, image):
        image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
        mask = torch.from_numpy(image_np)
        return 1.0 - mask
    
    def calculate_crop_area(self, image_width, image_height, crop_x, crop_y, crop_width, crop_height, padding=0):
        # Add padding to the crop area
        crop_x -= padding
        crop_y -= padding
        crop_width += 2 * padding
        crop_height += 2 * padding
        
        # Ensure the crop area is square by using the larger dimension
        final_size = max(crop_width, crop_height)
        
        # Adjust x and y to keep the crop within the full image boundaries
        crop_x = max(0, min(crop_x, image_width - final_size))
        crop_y = max(0, min(crop_y, image_height - final_size))
        
        # Make sure the crop size fits within the image boundaries
        if crop_x + final_size > image_width:
            crop_x = image_width - final_size
        if crop_y + final_size > image_height:
            crop_y = image_height - final_size

        # Ensure x, y, width, and height are integers
        crop_x = int(crop_x)
        crop_y = int(crop_y)
        final_size = int(final_size)
        
        return crop_x, crop_y, final_size, final_size

    def mask_crop_region(self, mask, padding=24, region_type="dominant"):
        mask_pil = Image.fromarray(np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        region_mask, crop_data = self.crop_region(mask_pil, region_type, padding)
        region_tensor = self.pil2mask(ImageOps.invert(region_mask)).unsqueeze(0).unsqueeze(1)

        (width, height), (left, top, right, bottom) = crop_data

        return (region_tensor, crop_data, top, left, right, bottom, width, height)

    def crop_region(self, mask, region_type, padding=0):
        
        from scipy.ndimage import label, find_objects
        
        binary_mask = np.array(mask.convert("L")) > 0
        bbox = mask.getbbox()
        if bbox is None:
            return mask, (mask.size, (0, 0, 0, 0))

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        # Calculate the crop area using the provided calculate_crop_area method
        crop_x, crop_y, crop_width, crop_height = self.calculate_crop_area(
            mask.width, mask.height, bbox[0], bbox[1], bbox_width, bbox_height, padding
        )

        cropped_mask = mask.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
        crop_data = (cropped_mask.size, (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

        return cropped_mask, crop_data
