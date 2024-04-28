#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import torch
import math
import numpy as np

import nodes
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
            (quarter_width, half_height, half_width, half_height)  # bottom middle

            (0, quarter_height, half_width, half_height),  # middle left
            (half_width, quarter_height, half_width, half_height),  # middle right
            (quarter_width, quarter_height, half_width, half_height),  # middle middle
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
    def rebuild_image_from_parts(self, output_images, origin_image, destination_image, feather_mask):
        
        original_width = origin_image.shape[2]
        original_height = origin_image.shape[1]
        channel_count = origin_image.shape[3]
        
        full_image = torch.zeros((origin_image.shape[0], original_height, original_width, channel_count), dtype=output_images[0].dtype, device=output_images[0].device)

        grid_specs = self.get_grid_specs(original_width, original_height)

        grid_mask = comfy_extras.nodes_mask.SolidMask.solid(comfy_extras.nodes_mask.SolidMask, 1, output_images[0].shape[2], output_images[0].shape[1])[0]
        grid_feathermask_vertical = comfy_extras.nodes_mask.FeatherMask.feather(comfy_extras.nodes_mask.FeatherMask, grid_mask, feather_mask, 0, feather_mask, 0)
        # grid_feathermask_horizontal = comfy_extras.nodes_mask.FeatherMask.feather(comfy_extras.nodes_mask.FeatherMask, grid_mask, 0, feather_mask, 0, feather_mask)
        grid_destination = torch.zeros((output_images[0].shape), dtype=output_images[0].dtype, device=output_images[0].device)

        for output_image, (x_start, y_start, width_inc, height_inc) in zip(output_images, grid_specs):
            _output_image = comfy_extras.nodes_upscale_model.ImageCompositeMasked.composite(comfy_extras.nodes_upscale_model.ImageCompositeMasked, grid_destination, output_image, x = 0, y = 0, resize_source = False, mask = grid_feathermask_vertical)
            full_image[:, y_start:y_start + height_inc, x_start:x_start + width_inc] = _output_image
            
        return full_image
