#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import torch
import torch.nn.functional as F

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
            width = round(ow*ratio)
            height = round(oh*ratio)
        outputs = mask.unsqueeze(1)
        outputs = F.interpolate(outputs, size=(height, width), mode="nearest")
        outputs = outputs.squeeze(1)

        return(outputs, outputs.shape[2], outputs.shape[1],)
    
    def calculate_crop_area(self, image_width, image_height, crop_x, crop_y, crop_width, crop_height):
        # Add padding to the crop area
        padding = 15
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
