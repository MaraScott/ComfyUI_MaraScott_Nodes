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
    def get_dynamic_grid_specs(self, width, height, rows_qty = 3, cols_qty = 3, size_unit = 64):
        
        tile_width = width // cols_qty        
        tile_height = height // rows_qty        
        tile_width_units_qty = math.floor(tile_width // size_unit)
        tile_height_units_qty = math.floor(tile_height // size_unit)
        tile_order_rows = MS_Array.reorder_edges_to_center(list(range(rows_qty)))
        tile_order_cols = MS_Array.reorder_edges_to_center(list(range(cols_qty)))
        _tile_width = (tile_width_units_qty + 1) * size_unit
        _tile_height = (tile_height_units_qty + 1) * size_unit
                        
        tiles = []
        for row_index, row in enumerate(tile_order_rows):
            for col_index, col in enumerate(tile_order_cols):
                index = (col * len(tile_order_rows)) + row
                x = col_index * tile_width_units_qty * size_unit
                if col_index == (cols_qty - 1):
                    x = x - size_unit
                y = row_index * tile_height_units_qty * size_unit
                if row_index == (rows_qty - 1):
                    y = y - size_unit
                tiles.append([
                    row_index, 
                    col_index, 
                    index,
                    x, # x 
                    y, # y
                    _tile_width, # width 
                    _tile_height, # height 
                ])
        return tiles, tile_width_units_qty, tile_height_units_qty, tile_width, tile_height
    
    @classmethod
    def get_grid_images(self, image, grid_specs):

        grids = [
            image[
                :,
                y_start:y_start + height_inc, 
                x_start:x_start + width_inc
            ] for _, _, _, x_start, y_start, width_inc, height_inc in grid_specs
        ]

        return grids

    @classmethod
    def rebuild_image_from_parts(self, iteration, output_images, upscaled, grid_specs, feather_mask):
        
        width_feather_seam = feather_mask
        height_feather_seam = feather_mask
        
        tile_width = output_images[0].shape[2]
        tile_height = output_images[0].shape[1]
            
        grid_mask = comfy_extras.nodes_mask.SolidMask().solid(1, tile_width, tile_height)[0]
        grid_feathermask_vertical = comfy_extras.nodes_mask.FeatherMask().feather( 
            grid_mask, 
            0, 
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
            0, 
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
        tile_qty = int((total ** 0.5))
        for index, grid_spec in enumerate(grid_specs):
            log(f"Rebuilding tile {index + 1}/{total}", None, None, f"Refining {iteration}")
            row, col, order, x_start, y_start, width_inc, height_inc = grid_spec
            tiles_order.append((order, output_images[index]))
            if col == 0:
                outputRow = nodes.ImagePadForOutpaint().expand_image(output_images[index], 0, 0, upscaled.shape[2] - tile_width, 0, 0)[0]
            elif col == (tile_qty - 1):
                _y_start = 0
                outputRow = comfy_extras.nodes_mask.ImageCompositeMasked().composite(outputRow, output_images[index], x = x_start, y = _y_start, resize_source = False, mask = grid_feathermask_vertical_right)[0]
            else:
                i = int(row - 1)
                _y_start = 0
                outputRow = comfy_extras.nodes_mask.ImageCompositeMasked().composite(outputRow, output_images[index], x = x_start, y = _y_start, resize_source = False, mask = grid_feathermask_vertical)[0]

            if col == (tile_qty - 1):
                if row == 0:
                    outputTopRow = [y_start, outputRow]
                elif row == (tile_qty - 1):
                    outputBottomRow = [y_start, outputRow]
                else:
                    i = int(row - 1)
                    outputMiddleRow.append([y_start, outputRow])
                    
        nb_middle_tiles = len(outputMiddleRow)
        image_height = upscaled.shape[1] - tile_height
        full_image = nodes.ImagePadForOutpaint().expand_image(outputTopRow[1], 0, 0, 0, image_height, 0)[0]
        if outputBottomRow[0] is not None:
            _y_start = outputBottomRow[0]
            full_image = comfy_extras.nodes_mask.ImageCompositeMasked().composite(full_image, outputBottomRow[1], x = 0, y = _y_start, resize_source = False, mask = grid_feathermask_horizontal_bottom)[0]
        if nb_middle_tiles > 0:
            for index, output in enumerate(outputMiddleRow):
                _y_start = output[0]
                full_image = comfy_extras.nodes_mask.ImageCompositeMasked().composite(full_image, output[1], x = 0, y = _y_start, resize_source = False, mask = grid_feathermask_horizontal)[0]
        
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

