#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import math
import numpy as np

import nodes

from ...utils.log import *

class Image:

    @classmethod
    def is_divisible_by_8(self, image):
        width, height = image.shape[1], image.shape[0]
        divisible_by_8 = (width % 8 == 0) and (height % 8 == 0)
        return divisible_by_8
        
    @classmethod
    def calculate_new_dimensions(self, image_width, image_height):
        def round_up_to_nearest_8(x):
            return math.ceil(x / 8) * 8
        new_width = round_up_to_nearest_8(image_width)
        new_height = round_up_to_nearest_8(image_height)
        return new_width, new_height
        
    @classmethod
    def get_grid_images(self, image):
        
        width = image.shape[2]
        height = image.shape[1]
        half_width = width // 2
        half_height = height // 2
        quarter_width = width // 4
        quarter_height = height // 4

        # Define the starting points and sizes for each grid section
        grid_specs = [
            (0, 0, quarter_width, quarter_height),  # top left
            (quarter_width, 0, quarter_width, quarter_height),  # top middle
            (half_width, 0, quarter_width, quarter_height),  # top right
            (0, quarter_height, quarter_width, quarter_height),  # middle left
            (quarter_width, quarter_height, quarter_width, quarter_height),  # center
            (half_width, quarter_height, quarter_width, quarter_height),  # middle right
            (0, half_height, quarter_width, quarter_height),  # bottom left
            (quarter_width, half_height, quarter_width, quarter_height),  # bottom middle
            (half_width, half_height, quarter_width, quarter_height)  # bottom right
        ]

        # Extract each grid section based on specified dimensions and starting points
        grids = [
            image[
                :, 
                y_start:y_start + height_inc, 
                x_start:x_start + width_inc
            ] 
            for (x_start, y_start, width_inc, height_inc) in grid_specs
        ]

        return grids
        
    @classmethod
    def rebuild_image_from_parts(self, output_images, origin_image):

        original_width = origin_image.shape[2]
        original_height = origin_image.shape[1]

        # Create an empty array to hold the full image
        full_image = np.zeros((original_height, original_width, output_images[0].shape[2]), dtype=output_images[0].dtype)

        # Define the start points and sizes for placing each grid section back
        grid_specs = [
            (0, 0, original_width // 4, original_height // 4),  # top left
            (original_width // 4, 0, original_width // 4, original_height // 4),  # top middle
            (original_width // 2, 0, original_width // 4, original_height // 4),  # top right
            (0, original_height // 4, original_width // 4, original_height // 4),  # middle left
            (original_width // 4, original_height // 4, original_width // 4, original_height // 4),  # center
            (original_width // 2, original_height // 4, original_width // 4, original_height // 4),  # middle right
            (0, original_height // 2, original_width // 4, original_height // 4),  # bottom left
            (original_width // 4, original_height // 2, original_width // 4, original_height // 4),  # bottom middle
            (original_width // 2, original_height // 2, original_width // 4, original_height // 4)  # bottom right
        ]

        # Place each grid section back into the appropriate position
        for output_image, (x_start, y_start, width_inc, height_inc) in zip(output_images, grid_specs):
            full_image[y_start:y_start + height_inc, x_start:x_start + width_inc, :] = output_image[:, :, :]

        return full_image

        