#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import torch
import math
import numpy as np
from PIL import Image

import nodes
import comfy
import comfy_extras

from .array import MS_Array

from ...utils.log import *

class MS_Image:
    
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
    def format_2_divby8(self, image):

        if image is None:
            raise ValueError("MaraScott Image: No image provided")

        if not isinstance(image, torch.Tensor):
            raise ValueError("MaraScott Image: Image provided is not a Tensor")
        
        width = image.shape[2]
        height = image.shape[1]
        is_dviby8 = self.is_divisible_by_8(image)
        if not is_dviby8:
            is_dviby8 = False
            width, height = self.calculate_new_dimensions(width, height)
            image = nodes.ImageScale.upscale(nodes.ImageScale, image, "nearest-exact", width, height, "center")[0]

        return image, width, height, is_dviby8


    @classmethod
    def get_dynamic_grid_specs(self, width, height, tile_rows = 3, tile_cols = 3, tile_size = 512, size_unit = 64):
        
        width_unit = height_unit = math.floor(tile_size // size_unit)
        tile_rows = width // (width_unit * size_unit)
        tile_cols = height // (height_unit * size_unit)
        tile_order_rows = MS_Array.reorder_edges_to_center(list(range(tile_rows)))
        tile_order_cols = MS_Array.reorder_edges_to_center(list(range(tile_cols)))
        width_unit_qty = width_unit + 1
        height_unit_qty = height_unit + 1
        tile_width = width_unit_qty * size_unit
        tile_height = height_unit_qty * size_unit
        
        tiles = []
        feather_mask = 64
        for col_index, col in enumerate(tile_order_cols):
            for row_index, row in enumerate(tile_order_rows):
                tiles.append([
                    row_index, 
                    col_index, 
                    (col * len(tile_order_rows)) + row,
                    (row * (width_unit * size_unit)) - (row * feather_mask), # x 
                    (col * (height_unit * size_unit)) - (col * feather_mask), # y
                    tile_width, # width 
                    tile_height, # height 
                ])
                        
        return tiles, width_unit, height_unit, tile_width, tile_height, tile_rows, tile_cols
    
    @classmethod
    def get_grid_images(self, image, rows = 3, cols = 3, tile_size = 512):

        width, height = image.shape[2], image.shape[1]
        
        grid_specs = self.get_dynamic_grid_specs(width, height, rows, cols, tile_size)[0]
            
        grids = [
            image[
                :,
                y_start:y_start + height_inc, 
                x_start:x_start + width_inc
            ] for _, _, _, x_start, y_start, width_inc, height_inc in grid_specs
        ]

        return grids

    @classmethod
    def rebuild_image_from_parts(self, iteration, output_images, upscaled_image, feather_mask = 16, rows = 3, cols = 3):
        
        upscaled_width = upscaled_image.shape[2]
        upscaled_height = upscaled_image.shape[1]
        channel_count = upscaled_image.shape[3]

        grid_specs, width_unit, height_unit, tile_width, tile_height, tile_rows, tile_cols = self.get_dynamic_grid_specs(upscaled_width, upscaled_height, rows, cols, 1024)
        
        width_feather_seam = feather_mask
        height_feather_seam = feather_mask
            
        grid_mask = comfy_extras.nodes_mask.SolidMask().solid(1, tile_width, tile_height)[0]
        grid_feathermask_vertical = comfy_extras.nodes_mask.FeatherMask().feather( 
            grid_mask, 
            width_feather_seam, 
            0, 
            width_feather_seam, 
            0
        )[0]
        grid_feathermask_vertical_right = comfy_extras.nodes_mask.FeatherMask().feather( 
            grid_mask, 
            width_feather_seam, 
            0, 
            0, 
            0
        )[0]
        grid_feathermask_horizontal = comfy_extras.nodes_mask.FeatherMask().feather(
            grid_mask, 
            0, 
            height_feather_seam, 
            0, 
            height_feather_seam
        )[0]
        grid_feathermask_horizontal_bottom = comfy_extras.nodes_mask.FeatherMask().feather(
            grid_mask, 
            0, 
            height_feather_seam, 
            0, 
            0
        )[0]

        index = 0
        total = len(output_images)
        
        tiles_order = []        
        outputTopRow = None
        outputBottomRow = None
        outputMiddleRow = []
        for index, grid_spec in enumerate(grid_specs):
            log(f"Rebuilding tile {index + 1}/{total}", None, None, f"Refining {iteration}")
            row, col, order, x_start, y_start, width_inc, height_inc = grid_spec
            tiles_order.append((order, output_images[index]))
            
            feather_qty = 3
            tile_qty = int((total ** 0.5))
            tiles_qty = tile_qty - 1
            if row == 0:
                outputRow = nodes.ImagePadForOutpaint().expand_image(output_images[index], 0, 0, (tiles_qty * tile_width) - (tiles_qty * feather_qty * width_feather_seam), 0, 0)[0]
            elif row == 1:
                y_start = 0 if not index == 2 else y_start
                x_start = x_start - (feather_qty * width_feather_seam)
                outputRow = comfy_extras.nodes_mask.ImageCompositeMasked().composite(outputRow, output_images[index], x = x_start, y = y_start, resize_source = False, mask = grid_feathermask_vertical_right)[0]
            else:
                y_start = 0 if not index == 1 else y_start
                x_start = x_start - width_feather_seam
                outputRow = comfy_extras.nodes_mask.ImageCompositeMasked().composite(outputRow, output_images[index], x = x_start, y = y_start, resize_source = False, mask = grid_feathermask_vertical)[0]

            if col == 0:
                outputTopRow = outputRow
            elif col == 1:
                outputBottomRow = outputRow
            else:
                i = int(col - 2)
                if len(outputMiddleRow) > i and outputMiddleRow[i] is not None:
                    outputMiddleRow[i] = outputRow
                else:
                    outputMiddleRow.append(outputRow)
                    
        nb_middle_tiles = len(outputMiddleRow)        
        full_image = nodes.ImagePadForOutpaint().expand_image(outputTopRow, 0, 0, 0, (tiles_qty * tile_height) - (tiles_qty * feather_qty * height_feather_seam), 0)[0]
        if outputBottomRow is not None:
            full_image = comfy_extras.nodes_mask.ImageCompositeMasked().composite(full_image, outputBottomRow, x = 0, y = ((nb_middle_tiles * tile_height) + tile_height) - (feather_qty * (nb_middle_tiles + 1)  * height_feather_seam), resize_source = False, mask = grid_feathermask_horizontal_bottom)[0]
        if nb_middle_tiles > 0:
            for index, output in enumerate(outputMiddleRow):
                i = index + 1
                full_image = comfy_extras.nodes_mask.ImageCompositeMasked().composite(full_image, output, x = 0, y = (i * tile_height) - (feather_qty * i * height_feather_seam), resize_source = False, mask = grid_feathermask_horizontal)[0]
        
        return full_image, tiles_order

    @staticmethod
    def empty(width, height):
        return torch.zeros((height, width, 3), dtype=torch.float32)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor)  -> Image:
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def pil2tensor(image:Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

