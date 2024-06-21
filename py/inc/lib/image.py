#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import comfy_extras.nodes_images
import torch
import torch.nn.functional as F
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

    @staticmethod
    def empty(width, height):
        return torch.zeros((height, width, 3), dtype=torch.float32)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor)  -> Image:
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def pil2tensor(image:Image) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    
class MS_Image_v2(MS_Image):

    @classmethod
    def get_dynamic_grid_specs(self, width, height, rows_qty = 3, cols_qty = 3, size_unit = 64):
        feather_size = size_unit
        if size_unit == 0:
            size_unit = 1 
        # size_unit AKA feather_size
        tile_width = width / cols_qty
        tile_height = height / rows_qty
        tile_width_units_qty = math.ceil(tile_width / size_unit)
        tile_height_units_qty = math.ceil(tile_height / size_unit)
        new_width = tile_width_units_qty * size_unit * cols_qty
        new_height = tile_height_units_qty * size_unit * rows_qty
        last_tile_width_diff = new_width - width
        last_tile_height_diff = new_height - height
        tile_order_rows = MS_Array.reorder_edges_to_center(list(range(rows_qty)))
        tile_order_cols = MS_Array.reorder_edges_to_center(list(range(cols_qty)))
                        
        tiles = []
        for row_index, row in enumerate(tile_order_rows):
            for col_index, col in enumerate(tile_order_cols):
                order = (row_index * len(tile_order_rows)) + col_index
                
                _tile_width = (tile_width_units_qty + 2) * size_unit
                _tile_height = (tile_height_units_qty + 2) * size_unit
                x_tile_coordinate = (col_index * tile_width_units_qty * size_unit)
                y_tile_coordinate = (row_index * tile_height_units_qty * size_unit)
                
                # if first or last width tile
                if col_index == 0:
                    x = x_tile_coordinate - (0 * feather_size)
                elif col_index == (cols_qty - 1):
                    x = x_tile_coordinate - (2 * feather_size) - last_tile_width_diff
                else:
                    x = x_tile_coordinate - (1 * feather_size)

                # if first or last height tile
                if row_index == 0:
                    y = y_tile_coordinate - (0 * feather_size)
                elif row_index == (rows_qty - 1):
                    y = y_tile_coordinate - (2 * feather_size) - last_tile_height_diff
                else:
                    y = y_tile_coordinate - (1 * feather_size)
                                                
                tiles.append([
                    row_index, 
                    col_index, 
                    order,
                    x, # x 
                    y, # y
                    _tile_width, # width 
                    _tile_height, # height 
                ])
        return tiles, tile_width_units_qty, tile_height_units_qty, tile_width, tile_height
    
    @classmethod
    def get_tiled_grid_specs(self, image, tile_size, rows_qty = 3, cols_qty = 3, size_unit = 64):
        width = tile_size
        height = tile_size
        feather_size = size_unit
        if size_unit == 0:
            size_unit = 1 
        # size_unit AKA feather_size
        tile_width = width
        tile_height = height
        tile_width_units_qty = math.ceil(tile_width / size_unit)
        tile_height_units_qty = math.ceil(tile_height / size_unit)
        tile_order_rows = MS_Array.reorder_edges_to_center(list(range(rows_qty)))
        tile_order_cols = MS_Array.reorder_edges_to_center(list(range(cols_qty)))
                        
        tiles = []
        for row_index, row in enumerate(tile_order_rows):
            for col_index, col in enumerate(tile_order_cols):
                order = (row_index * len(tile_order_rows)) + col_index
                
                _tile_width = (tile_width_units_qty + 2) * size_unit
                _tile_height = (tile_height_units_qty + 2) * size_unit
                x_tile_coordinate = (col_index * tile_size)
                y_tile_coordinate = (row_index * tile_size)
                                
                # if first or last width tile
                if col_index == 0:
                    x = x_tile_coordinate - (0 * feather_size)
                elif col_index == (cols_qty - 1):
                    x = image.shape[2] - _tile_width
                else:
                    x = x_tile_coordinate - (1 * feather_size)
                    if x + _tile_width > image.shape[2]:
                        x = x - ((x + _tile_width) - image.shape[2])

                # if first or last height tile
                if row_index == 0:
                    y = y_tile_coordinate - (0 * feather_size)
                elif row_index == (rows_qty - 1):
                    y = image.shape[1] - _tile_height
                else:
                    y = y_tile_coordinate - (1 * feather_size)
                    if y + _tile_height > image.shape[1]:
                        y = y - ((y + _tile_height) - image.shape[1])

                tiles.append([
                    row_index, 
                    col_index, 
                    order,
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
                x_start:x_start + width_inc,
                :
            ] for _, _, _, x_start, y_start, width_inc, height_inc in grid_specs
        ]

        return grids
    
    @classmethod
    def rebuild_image_from_parts(self, iteration, output_images, image, grid_specs, feather_mask, upscale_scale, rows_qty, cols_qty, grid_prompts):
        
        width_feather_seam = feather_mask
        height_feather_seam = feather_mask
        
        tile_width = output_images[0].shape[2]
        tile_height = output_images[0].shape[1]
            
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
        last_tile_row_index = max(grid_spec[0] for grid_spec in grid_specs)
        last_tile_col_index = max(grid_spec[1] for grid_spec in grid_specs)
        
        tiles_order = []        
        outputTopRow = [None, None]
        outputBottomRow = [None, None]
        outputMiddleRow = []
        for index, grid_spec in enumerate(grid_specs):
            log(f"Rebuilding tile {index + 1}/{total}", None, None, f"Refining {iteration}")
            row, col, order, x_start, y_start, width_inc, height_inc = grid_spec
            prompt = grid_prompts[index] if 0 <= index < len(grid_prompts) else ""
            tiles_order.append((order, output_images[index], prompt))
            if col == 0:
                outputRow = nodes.ImagePadForOutpaint().expand_image(output_images[index], 0, 0, ((cols_qty - 1) * tile_width), 0, 0)[0]
            elif col == last_tile_col_index:
                _y_start = 0
                outputRow = comfy_extras.nodes_mask.ImageCompositeMasked().composite(outputRow, output_images[index], x = x_start, y = _y_start, resize_source = False, mask = grid_feathermask_vertical_right)[0]
            else:
                i = int(row - 1)
                _y_start = 0
                outputRow = comfy_extras.nodes_mask.ImageCompositeMasked().composite(outputRow, output_images[index], x = x_start, y = _y_start, resize_source = False, mask = grid_feathermask_vertical)[0]

            if col == last_tile_col_index:
                if row == 0:
                    outputTopRow = [y_start, outputRow]
                elif row == last_tile_row_index:
                    outputBottomRow = [y_start, outputRow]
                else:
                    i = int(row - 1)
                    outputMiddleRow.append([y_start, outputRow])
                    
        nb_middle_tiles = len(outputMiddleRow)
        image_height = ((rows_qty - 1) * tile_height)
        full_image = nodes.ImagePadForOutpaint().expand_image(outputTopRow[1], 0, 0, 0, image_height, 0)[0]
        if outputBottomRow[0] is not None:
            _y_start = outputBottomRow[0]
            full_image = comfy_extras.nodes_mask.ImageCompositeMasked().composite(full_image, outputBottomRow[1], x = 0, y = _y_start, resize_source = False, mask = grid_feathermask_horizontal_bottom)[0]
        if nb_middle_tiles > 0:
            for index, output in enumerate(outputMiddleRow):
                _y_start = output[0]
                full_image = comfy_extras.nodes_mask.ImageCompositeMasked().composite(full_image, output[1], x = 0, y = _y_start, resize_source = False, mask = grid_feathermask_horizontal)[0]
        
        full_image = comfy_extras.nodes_images.ImageCrop().crop(full_image, (image.shape[2] * upscale_scale), (image.shape[1] * upscale_scale), 0, 0)[0]
        
        return full_image, tiles_order

class MS_Image_v1(MS_Image):
    
    @classmethod
    def get_dynamic_grid_specs(self, width, height, tile_rows = 3, tile_cols =3):
        
        width_unit = width // 16
        height_unit = height // 16
        tile_width = width_unit * 6
        tile_height = height_unit * 6
                
        tiles = []
        tile_order = [0,2,1]
        for col in tile_order:
            for row in tile_order:
                tiles.append([
                    (col * len(tile_order)) + row,
                    (row * tile_width) - (row * width_unit), # x 
                    (col * tile_height) - (col * height_unit), # y 
                    tile_width, # width 
                    tile_height, # height 
                ])
                
        return tiles, width_unit, height_unit, tile_width, tile_height
    
    @classmethod
    def get_9_grid_specs(self, width, height):

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
    def get_grid_images(self, image, rows = 3, cols = 3):
        width, height = image.shape[2], image.shape[1]
        
        grid_specs = self.get_dynamic_grid_specs(width, height, rows, cols)[0]
            
        grids = [
            image[
                :,
                y_start:y_start + height_inc, 
                x_start:x_start + width_inc
            ] for _, x_start, y_start, width_inc, height_inc in grid_specs
        ]

        return grids

    @classmethod
    def rebuild_image_from_parts(self, iteration, output_images, upscaled_image, feather_mask = 16, rows = 3, cols = 3):
        
        upscaled_width = upscaled_image.shape[2]
        upscaled_height = upscaled_image.shape[1]
        channel_count = upscaled_image.shape[3]

        grid_specs, width_unit, height_unit, tile_width, tile_height = self.get_dynamic_grid_specs(upscaled_width, upscaled_height, rows, cols)
        
        width_feather_seam = feather_mask
        height_feather_seam = feather_mask
            
        grid_mask = comfy_extras.nodes_mask.SolidMask.solid(comfy_extras.nodes_mask.SolidMask, 1, tile_width, tile_height)[0]
        grid_feathermask_vertical = comfy_extras.nodes_mask.FeatherMask.feather(
            comfy_extras.nodes_mask.FeatherMask, 
            grid_mask, 
            width_feather_seam, 
            0, 
            width_feather_seam, 
            0
        )[0]
        grid_feathermask_horizontal = comfy_extras.nodes_mask.FeatherMask.feather(
            comfy_extras.nodes_mask.FeatherMask, 
            grid_mask, 
            0, 
            height_feather_seam, 
            0, 
            height_feather_seam
        )[0]

        index = 0
        total = len(output_images)
        tiles_order = []

        for index, grid_spec in enumerate(grid_specs):
            log(f"Rebuilding tile {index + 1}/{total}", None, None, f"Refining {iteration}")
            order, x_start, y_start, width_inc, height_inc = grid_spec
            tiles_order.append((order, output_images[index]))
            if index in [0,3,6]:
                outputRow = nodes.ImagePadForOutpaint.expand_image(nodes.ImagePadForOutpaint, output_images[index], 0, 0, (2 * tile_width) - (2 * width_unit), 0, 0)[0]
            if index in [1,4,7]:
                if not index == 1:
                    y_start = 0
                outputRow = comfy_extras.nodes_mask.ImageCompositeMasked().composite(outputRow, output_images[index], x = x_start, y = y_start, resize_source = False, mask = None)[0]
            if index in [2,5,8]:
                if not index == 2:
                    y_start = 0
                outputRow = comfy_extras.nodes_mask.ImageCompositeMasked().composite(outputRow, output_images[index], x = x_start, y = y_start, resize_source = False, mask = grid_feathermask_vertical)[0]
            if index in [0,1,2]:
                outputTopRow = outputRow
            if index in [3,4,5]:
                outputBottomRow = outputRow
            if index in [6,7,8]:
                outputMiddleRow = outputRow
                
        full_image = nodes.ImagePadForOutpaint().expand_image(outputTopRow, 0, 0, 0, (2 * tile_height) - (2 * height_unit), 0)[0]
        full_image = comfy_extras.nodes_mask.ImageCompositeMasked().composite(full_image, outputBottomRow, x = 0, y = (2 * tile_height) - (2 * height_unit), resize_source = False, mask = None)[0]
        full_image = comfy_extras.nodes_mask.ImageCompositeMasked().composite(full_image, outputMiddleRow, x = 0, y = (1 * tile_height) - (1 * height_unit), resize_source = False, mask = grid_feathermask_horizontal)[0]
        
        return full_image, tiles_order
