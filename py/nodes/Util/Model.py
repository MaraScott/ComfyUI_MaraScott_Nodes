import torch
import torchvision.transforms as transforms
import re
from collections import defaultdict
import random
from PIL import Image, ImageDraw
import numpy as np

from ...inc.lib.image import MS_Image
from ...utils.constants import get_name, get_category
from ...utils.helper import AlwaysEqualProxy, is_user_defined_object, natural_key
from ...utils.log import log

any_type = AlwaysEqualProxy("*")

class GetModelBlocks_v1:

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
                
class GetModelBlocksHeatmap_v1:

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
        # Create a blank heatmap image at the desired size
        heatmap = Image.new("RGB", (heatmap_width, heatmap_height), (255, 255, 255))
        draw = ImageDraw.Draw(heatmap)

        # Define layers and variations
        total_layers = 27  # 12 input layers + 3 middle layers + 12 output layers
        initial_variations = 14

        # Create a smooth gradient from lighter dark gray (top-left) to light gray (bottom-right)
        for y in range(heatmap_height):
            for x in range(heatmap_width):
                # Calculate intensity based on position (linear gradient from top-left to bottom-right)
                intensity = 200 - int((x + y) / (heatmap_width + heatmap_height) * 100)  # Lighter dark gray, intensity range 200 to 100
                intensity = max(0, min(255, intensity))  # Clamp the intensity between 0 and 255
                color = (intensity, intensity, intensity)
                draw.point((x, y), fill=color)
                
        # Create a table of all variations to use for coloring pixels
        variation_table = []
        for input_layer in range(12):
            variation_table.append(f'input_blocks.{input_layer}={variation_min}')
        for middle_layer in range(3):
            variation_table.append(f'middle_blocks.{middle_layer}={variation_min}')
        for output_layer in range(12):
            variation_table.append(f'output_blocks.{output_layer}={variation_min}')
        variation_table.append(f'variation_range={variation_min}-{variation_max}')                

        # Determine the number of variations in the specified range
        total_variations_in_range = int((variation_max - variation_min) * variation_qty)

        # Randomly sample variations within the specified range to overlay with colorful dots
        sampled_variations = random.sample(range(total_variations_in_range), max_variations)
        sampled_positions = set()

        # Overlay the sampled variations with colorful dots
        for idx in sampled_variations:
            # Randomly select x and y positions within the heatmap size
            x = random.randint(0, heatmap_width - 10)
            y = random.randint(0, heatmap_height - 10)
            sampled_positions.add((x, y))

            # Determine color based on the variation index to distinguish different types of layers
            layer_idx = idx % total_layers
            if layer_idx < 12:
                color = (0, 0, 255)  # Blue for input layers
            elif layer_idx < 15:
                color = (255, 165, 0)  # Orange for middle layers
            else:
                color = (255, 0, 0)  # Red for output layers

            # Draw a small dot to represent the sampled variation
            draw.ellipse(
                [x, y, x + 10, y + 10],  # Size of the dot representing the variation
                fill=color
            )

        # Convert the heatmap image to a tensor
        heatmap_tensor = MS_Image.pil2tensor(heatmap)
        return heatmap_tensor
