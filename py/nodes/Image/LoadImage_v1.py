import os
import hashlib
import nodes
import folder_paths

from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch


from ...utils.constants import get_name, get_category
from ...utils.log import log
from ...utils.helper import current_method

from inspect import cleandoc, currentframe as cf


class nodes_LoadImage_v1(nodes.LoadImage):
    
    """
    --- test doc ---
    """

    def __init__(self):
        super().__init__()
        
    CATEGORY = get_category("Image")

    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(s):

        input_dir = folder_paths.get_input_directory()
        exclude_folders = []
        file_list = []
        for root, dirs, files in os.walk(input_dir):
            # Exclude specific folders
            dirs[:] = [d for d in dirs if d not in exclude_folders]

            for file in files:
                if os.path.splitext(file)[1].lower() in s.ALLOWED_EXTENSIONS:
                    relpath = os.path.relpath(os.path.join(root, file), start=input_dir)
                    # fix for windows
                    relpath = relpath.replace("\\", "/")
                    file_list.append(relpath)

        return {
            "required": {
                "image": (sorted(file_list), {"image_upload": True})
            },
        }

    @classmethod
    def _INPUT_TYPES(s, EXTRA_INPUT_TYPES):
        _input_types = getattr(super(), "INPUT_TYPES", lambda: {})()

        for key, value in EXTRA_INPUT_TYPES.items():
            if key in _input_types:
                if isinstance(_input_types[key], dict) and isinstance(value, dict):
                    _input_types[key].update(value)
                else:
                    _input_types[key] = value
            else:
                _input_types[key] = value

        return _input_types

    @classmethod
    def _RETURN_OUTPUTS(s, OUTPUT_PARAM_NAME):
        OUTPUT_PARAMS = getattr(super(), OUTPUT_PARAM_NAME, ())
        if OUTPUT_PARAM_NAME == "RETURN_NAMES" and len(OUTPUT_PARAMS) == 0:
            OUTPUT_PARAMS = getattr(super(), "RETURN_TYPES", ())
        if not isinstance(OUTPUT_PARAMS, tuple):
            OUTPUT_PARAMS = (OUTPUT_PARAMS,)
        return OUTPUT_PARAMS

    @classmethod
    def IS_CHANGED(s, **kwargs):
        image = kwargs.get("image", None)
        return current_method(super(), cf())(image)

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        image = kwargs.get("image", None)
        return current_method(super(), cf())(image)


class LoadImage_v1(nodes_LoadImage_v1):

    def __init__(self):
        super().__init__()

    NAME = "Load Image"
    SHORTCUT = "i"

    @classmethod
    def _INPUT_TYPES(s):

        INPUT_TYPES = {
            "required": {
                "filename_with_ext": ("BOOLEAN", {"default": False}),
            },
            "hidden": {},
        }

        return super()._INPUT_TYPES(INPUT_TYPES)

    RETURN_TYPES = nodes_LoadImage_v1()._RETURN_OUTPUTS("RETURN_TYPES") + ("STRING",)
    RETURN_NAMES = nodes_LoadImage_v1()._RETURN_OUTPUTS("RETURN_NAMES") + ("File Name",)

    INPUT_IS_LIST = False
    FUNCTION = "fn"
    CATEGORY = get_category("image")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (None,) * len(RETURN_TYPES)

    def fn(s, **kwargs):
        image = kwargs.get("image", None)
        filename_with_ext = kwargs.get("filename_with_ext", None)

        file_path = folder_paths.get_annotated_filepath(image)
        try:
            output_image, output_mask = super().load_image(file_path)
        except IndexError:
            # 1) Open and fix orientation, convert to RGBA to preserve alpha if present
            img_pil = Image.open(file_path)
            img_pil = ImageOps.exif_transpose(img_pil).convert("RGBA")
            # 2) Into NumPy and normalize to [0,1]
            arr = np.array(img_pil).astype(np.float32) / 255.0  # shape (H, W, 4)
            # 3) Split RGB vs. alpha
            rgb = arr[..., :3]      # [H, W, 3]
            alpha = arr[..., 3]     # [H, W]
            # 4) Build image tensor [1, C, H, W]
            image_tensor = torch.from_numpy(rgb).unsqueeze(0)
            # 5) Build mask tensor [1, 1, H, W] with inverted alpha
            mask_tensor  = (1.0 - torch.from_numpy(alpha)).unsqueeze(0).unsqueeze(-1)
            output_image, output_mask = image_tensor, mask_tensor
        filename = file_path.replace(folder_paths.get_input_directory() + "\\", "")
        if not filename_with_ext:
            filename = os.path.splitext(filename)[0]

        return (
            output_image,
            output_mask,
            filename,
        )
