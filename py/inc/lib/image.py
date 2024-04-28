#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import torch
import math
import numpy as np

import nodes

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
            (quarter_width, 0, half_width, half_height),  # top middle
            (half_width, 0, half_width, half_height),  # top right
            (0, quarter_height, half_width, half_height),  # middle left
            (quarter_width, quarter_height, half_width, half_height),  # center
            (half_width, quarter_height, half_width, half_height),  # middle right
            (0, half_height, half_width, half_height),  # bottom left
            (quarter_width, half_height, half_width, half_height),  # bottom middle
            (half_width, half_height, half_width, half_height)  # bottom right
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
    def rebuild_image_from_parts(self, output_images, origin_image):
        original_width = origin_image.shape[2]
        original_height = origin_image.shape[1]
        channel_count = origin_image.shape[3]

        full_image = torch.zeros((origin_image.shape[0], original_height, original_width, channel_count), dtype=output_images[0].dtype, device=output_images[0].device)

        grid_specs = self.get_grid_specs(original_width, original_height)

        for output_image, (x_start, y_start, width_inc, height_inc) in zip(output_images, grid_specs):
            full_image[:, y_start:y_start + height_inc, x_start:x_start + width_inc] = output_image
            
        log(origin_image.shape)
        log(full_image.shape)
        log(full_image.shape)

        return full_image
