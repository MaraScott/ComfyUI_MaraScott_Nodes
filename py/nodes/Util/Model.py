import os
import torch
import torchvision.transforms as transforms
import re
from collections import defaultdict
import random
from PIL import Image, ImageDraw
import numpy as np
from itertools import product

from ...inc.lib.image import MS_Image
from ...utils.constants import get_name, get_category
from ...utils.helper import AlwaysEqualProxy, is_user_defined_object, natural_key
from ...utils.log import log

any_type = AlwaysEqualProxy("*")

class ModelBlocks_v1:

    @classmethod
    def get_blocks(self, model):
        blocks = []
        pattern = re.compile(r"(\w+\.\w+\.\d+)|(\w+\.\w+\[\d+\])")
        
        for key in model.model_state_dict().keys():
            if "block" in key:
                match = pattern.search(key)
                if match:
                    block = match.group()
                    if block.startswith("diffusion_model."):
                        block = block.replace("diffusion_model.", "", 1)
                    blocks.append(block)
                    
        return sorted(set(blocks), key=natural_key)
                
class GetModelBlocks_v1(ModelBlocks_v1):

    NAME = get_name('Get Model Blocks')

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "type": (["Names", "Variations"], { "default": "Names" }),
                "variation": ("FLOAT", { "label": "Variation (max)", "default": 1.4, "min": 0.1, "max": 14.0, "step":0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("BLOCKS", "NAMES", "COUNT")
    FUNCTION = "fn"
    CATEGORY = get_category('utils')

    @classmethod
    def fn(self, model=None, type="Names", variation=1.4):
        model_blocks_string = ""
        model_blocks_length = 0
        model_blocks_names_string = ""
        if is_user_defined_object(model):
            try:
                model_blocks, model_blocks_names = self.get_blocks_variations(model, type, variation)
                model_blocks_string = "\n".join(model_blocks)
                model_blocks_length = len(model_blocks)
                model_blocks_names_string = "\n".join(model_blocks_names)
            except Exception:
                log(str(model), None, None, "model_blocks error")
        return (model_blocks_string, model_blocks_names_string, model_blocks_length)

    @classmethod
    def get_blocks_variations(self, model=None, type="Names", variation=1.4):
        all_blocks = self.get_blocks(model)
        blocks = []
        blocks_names = []
        variations = int(variation*10)
        for block in all_blocks:
            block_name = block.replace(".", "_")
            block = block.replace(".", r"\.")
            if type == "Variations":            
                for i in range(1, int(variations+1)):
                    x = round(0.1 * i, 1)
                    block_var=f"{block}={x}"
                    blocks.append(block_var)
                    block_name_var=f"{block_name}_weight_{x}"
                    blocks_names.append(block_name_var)
            else:
                blocks.append(block)                
                blocks_names.append(block_name)
        return blocks, blocks_names
        
class GetModelBlocksHeatmap_v1(ModelBlocks_v1):
    
    # the number of variation for sd1.5 if you modify the weights is 8.82 x 10^27

    NAME = get_name('Get Model Blocks Heatmap')

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", { "label": "height", "default": 1024}),
                "height": ("INT", { "label": "height", "default": 1024}),
                "variation": ("FLOAT", { "label": "Variation (max)", "default": 1.4, "min": 0.1, "max": 14.0, "step":0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("HEATMAP",)
    FUNCTION = "fn"
    CATEGORY = get_category('utils')

    @classmethod
    def fn(self, model=None, variation=1.4, width=1024, height=1024):
                
        variation_qty = 1185024  # Simulate one million variations
        heatmap_size = 4096

        heatmap_tensor = self.generate_heatmap(
            variation_qty=variation_qty,
            img_width=width,
            img_height=height,
            heatmap_width=int(variation_qty**0.5),
            heatmap_height=int(variation_qty**0.5),
            max_variations=250,  # Limit to 250 variations for actual processing
            variation_min=0.8,
            variation_max=1.4
        )

        return (heatmap_tensor,)

    @classmethod
    def generate_heatmap(self, variation_qty, img_width, img_height, heatmap_width, heatmap_height, max_variations, variation_min, variation_max):

        # Define the structure and weights
        categories = {
            "input_blocks": range(2),    # 0 to 11
            "middle_blocks": range(3),    # 0 to 2
            "output_blocks": range(2)    # 0 to 11
        }
        weights = [0.1]
        
        # Generate all possible combinations of weights
        all_combinations = list(product(weights, repeat=sum(len(v) for v in categories.values())))
        
        # Calculate image dimensions based on number of combinations
        num_pixels = len(all_combinations)
        img_size = int(num_pixels ** 0.5) + 1  # Create a square image
        
        # Create a blank heatmap image at the desired size
        heatmap = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        pixels = heatmap.load()
        
        # Generate unique color for each variation and map it to pixel
        for i, combination in enumerate(all_combinations):
            # Create a unique RGB color for each combination
            r = int((combination[0] + combination[2] + combination[4]) * 100) % 256
            g = int((combination[1] + combination[3] + combination[5]) * 100) % 256
            b = (i * 50) % 256  # Ensure each combination has a unique color
            
            # Place the pixel in the image
            x, y = divmod(i, img_size)
            if x < img_size and y < img_size:  # Check bounds
                pixels[y, x] = (r, g, b)

        # Convert the heatmap image to a tensor
        heatmap_tensor = MS_Image.pil2tensor(heatmap)
        return heatmap_tensor

