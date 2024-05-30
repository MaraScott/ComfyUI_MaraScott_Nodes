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