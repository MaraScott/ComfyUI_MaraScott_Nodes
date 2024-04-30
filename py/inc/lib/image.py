#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import torch
import math
import numpy as np

import nodes
import comfy
import comfy_extras

from ...utils.log import *

class Image:
    
    @classmethod
    def is_divisible_by_8(self, image):
        width, height = image.shape[1], image.shape[0]
        return (width % 8 == 0) and (height % 8 == 0)

    @classmethod
    def calculate_new_dimensions(self, image_width, image_height):
        def round_up_to_nearest_8(x):
            return math.ceil(x / 8) * 8
        new_width = round_up_to_nearest_8(image_width)
        new_height = round_up_to_nearest_8(image_height)
        return new_width, new_height

    @classmethod
    def get_grid_specs(self, width, height):

        half_width = width // 2
        half_height = height // 2
        quarter_width = width // 4
        quarter_height = height // 4

        return [
            (0, 0, half_width, half_height),  # top left
            (half_width, 0, half_width, half_height),  # top right
            (quarter_width, 0, half_width, half_height),  # top middle

            (0, half_height, half_width, half_height),  # bottom left
            (half_width, half_height, half_width, half_height),  # bottom right
            (quarter_width, half_height, half_width, half_height),  # bottom middle

            (0, quarter_height, half_width, half_height),  # middle left
            (half_width, quarter_height, half_width, half_height),  # middle right
            (quarter_width, quarter_height, half_width, half_height)  # middle middle
        ]

    @classmethod
    def get_grid_images(self, image):
        width, height = image.shape[2], image.shape[1]
        
        grid_specs = self.get_grid_specs(width, height)

        grids = [
            image[
                :,
                y_start:y_start + height_inc, 
                x_start:x_start + width_inc
            ] for x_start, y_start, width_inc, height_inc in grid_specs
        ]

        return grids

    @classmethod
    def rebuild_image_from_parts(self, output_images, upscaled_image, feather_mask):
        
        upscaled_width = upscaled_image.shape[2]
        upscaled_height = upscaled_image.shape[1]
        channel_count = upscaled_image.shape[3]

        grid_specs = self.get_grid_specs(upscaled_width, upscaled_height)

        scale_by = upscaled_width // grid_specs[0][2]
        
        grid_mask = comfy_extras.nodes_mask.SolidMask.solid(comfy_extras.nodes_mask.SolidMask, 1, grid_specs[0][2], grid_specs[0][3])[0]
        grid_feathermask_vertical = comfy_extras.nodes_mask.FeatherMask.feather(comfy_extras.nodes_mask.FeatherMask, grid_mask, feather_mask, 0, feather_mask, 0)[0]
        grid_feathermask_horizontal = comfy_extras.nodes_mask.FeatherMask.feather(comfy_extras.nodes_mask.FeatherMask, grid_mask, 0, feather_mask, 0, feather_mask)[0]

        index = 0
        total = len(output_images)
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputTopRow = nodes.ImagePadForOutpaint.expand_image(nodes.ImagePadForOutpaint, output_images[index], 0, 0, width_inc, 0, 0)[0]
        index = index + 1
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputTopRow = comfy_extras.nodes_mask.ImageCompositeMasked.composite(comfy_extras.nodes_mask.ImageCompositeMasked, outputTopRow, output_images[index], x = x_start, y = y_start, resize_source = False, mask = None)[0]
        index = index + 1
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputTopRow = comfy_extras.nodes_mask.ImageCompositeMasked.composite(comfy_extras.nodes_mask.ImageCompositeMasked, outputTopRow, output_images[index], x = x_start, y = y_start, resize_source = False, mask = grid_feathermask_vertical)[0]

        index = index + 1
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputBottomRow = nodes.ImagePadForOutpaint.expand_image(nodes.ImagePadForOutpaint, output_images[index], 0, 0, width_inc, 0, 0)[0]
        index = index + 1
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputBottomRow = comfy_extras.nodes_mask.ImageCompositeMasked.composite(comfy_extras.nodes_mask.ImageCompositeMasked, outputBottomRow, output_images[index], x = x_start, y = 0, resize_source = False, mask = None)[0]
        index = index + 1
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputBottomRow = comfy_extras.nodes_mask.ImageCompositeMasked.composite(comfy_extras.nodes_mask.ImageCompositeMasked, outputBottomRow, output_images[index], x = x_start, y = 0, resize_source = False, mask = grid_feathermask_vertical)[0]
        
        index = index + 1
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputMiddleRow = nodes.ImagePadForOutpaint.expand_image(nodes.ImagePadForOutpaint, output_images[index], 0, 0, width_inc, 0, 0)[0]
        index = index + 1
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputMiddleRow = comfy_extras.nodes_mask.ImageCompositeMasked.composite(comfy_extras.nodes_mask.ImageCompositeMasked, outputMiddleRow, output_images[index], x = x_start, y = 0, resize_source = False, mask = None)[0]
        index = index + 1
        log(f"Rebuilding tile {index + 1}/{total}")
        x_start, y_start, width_inc, height_inc = grid_specs[index]
        outputMiddleRow = comfy_extras.nodes_mask.ImageCompositeMasked.composite(comfy_extras.nodes_mask.ImageCompositeMasked, outputMiddleRow, output_images[index], x = x_start, y = 0, resize_source = False, mask = grid_feathermask_vertical)[0]

        full_image = nodes.ImagePadForOutpaint.expand_image(nodes.ImagePadForOutpaint, outputTopRow, 0, 0, 0, outputTopRow.shape[1], 0)[0]
        full_image = comfy_extras.nodes_mask.ImageCompositeMasked.composite(comfy_extras.nodes_mask.ImageCompositeMasked, full_image, outputBottomRow, x = 0, y = outputBottomRow.shape[1], resize_source = False, mask = None)[0]
        full_image = comfy_extras.nodes_mask.ImageCompositeMasked.composite(comfy_extras.nodes_mask.ImageCompositeMasked, full_image, outputMiddleRow, x = 0, y = outputMiddleRow.shape[1] // 2, resize_source = False, mask = grid_feathermask_horizontal)[0]
        
        return full_image
