import re
from collections import defaultdict

from ...utils.constants import get_name, get_category
from ...utils.helper import AlwaysEqualProxy, is_user_defined_object, natural_key
from ...utils.log import log

any_type = AlwaysEqualProxy("*")

class GetModelBlocks_v1:

    NAME = get_name('Get Model Blocks', "m")

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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("BLOCKS", "NAMES", "BLOCKS WEIGTHS", "COUNT")
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")

    @classmethod
    def fn(self, model=None, type="Names", variation=1.4):
        model_blocks_string = ""
        model_blocks_length = 0
        model_blocks_names_string = ""
        if is_user_defined_object(model):
            try:
                model_blocks, model_blocks_names, model_block_weights = self.get_blocks_variations(model, type, variation)
                model_blocks_string = "\n".join(model_blocks)
                model_blocks_length = len(model_blocks)
                model_blocks_names_string = "\n".join(model_blocks_names)
                model_block_weights_string = "\n".join(model_block_weights)
            except Exception:
                log(str(model), None, None, "model_blocks error")
        return (model_blocks_string, model_blocks_names_string, model_block_weights_string, model_blocks_length)

    @classmethod
    def get_blocks_variations(self, model=None, type="Names", variation=1.4):
        all_blocks = self.get_blocks(model)
        blocks = []
        block_weights = []
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
                    block_weights_var=f"{block}=%WEIGHT%"
                    block_weights.append(block_weights_var)
                    block_name_var=f"{block_name}_weight_{x}"
                    blocks_names.append(block_name_var)
            else:
                blocks.append(block)
                block_weights_var=f"{block}=%WEIGHT%"
                block_weights.append(block_weights_var)
                blocks_names.append(block_name)
        return blocks, blocks_names, block_weights
        
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
                