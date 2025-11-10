import os
import hashlib
import nodes
import folder_paths
import requests
from io import BytesIO

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
                "load_mode": (["file_picker", "url_or_path"], {
                    "default": "file_picker",
                    "tooltip": "Choose how to specify the image: file picker from input directory or URL/path input"
                }),
            },
            "optional": {
                "url_or_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "URL (http/https) or file path (full or relative) to load image from. Only used when load_mode is 'url_or_path'."
                }),
            },
            "hidden": {},
        }

        return super()._INPUT_TYPES(INPUT_TYPES)

    RETURN_TYPES = nodes_LoadImage_v1()._RETURN_OUTPUTS("RETURN_TYPES") + ("INT", "INT", "STRING",)
    RETURN_NAMES = nodes_LoadImage_v1()._RETURN_OUTPUTS("RETURN_NAMES") + ("Width", "Height", "File Name",)

    INPUT_IS_LIST = False
    FUNCTION = "fn"
    CATEGORY = get_category("image")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (None,) * len(RETURN_TYPES)

    def fn(self, **kwargs):
        image = kwargs.get("image", None)
        filename_with_ext = kwargs.get("filename_with_ext", False)
        load_mode = kwargs.get("load_mode", "file_picker")
        url_or_path = kwargs.get("url_or_path", "")

        # Determine the file path based on load mode
        if load_mode == "url_or_path" and url_or_path.strip():
            file_path, actual_filename = self._load_from_url_or_path(url_or_path.strip())
        else:
            # Use file picker mode (original behavior)
            file_path = folder_paths.get_annotated_filepath(image)
            actual_filename = os.path.basename(image) if image else "unknown"

        # Load the image
        output_image, output_mask = self._load_image_from_path(file_path)

        # Get image dimensions
        height, width = output_image.shape[1:3]

        # Process filename for file picker mode or use the one from URL/path mode
        if load_mode == "file_picker":
            actual_filename = self._process_filename_for_file_picker(file_path, filename_with_ext)

        # Remove extension if requested (for both modes)
        if not filename_with_ext and actual_filename:
            actual_filename = os.path.splitext(actual_filename)[0]

        return (
            output_image,
            output_mask,
            width,
            height,
            actual_filename,
        )

    def _load_from_url_or_path(self, url_or_path: str):
        """Load image from URL or file path."""
        if url_or_path.startswith(('http://', 'https://')):
            # Load from URL
            log(f"Loading image from URL: {url_or_path}", None, None, "info")
            response = requests.get(url_or_path, stream=True, timeout=30)
            response.raise_for_status()

            # Save to temp directory for processing
            temp_dir = folder_paths.get_temp_directory()
            filename = url_or_path.split('/')[-1] or "downloaded_image"
            # Ensure it has an image extension
            if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']):
                filename += '.jpg'

            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return temp_path, filename
        else:
            # Load from file path
            resolved_path = self._resolve_file_path(url_or_path)
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"File not found: {resolved_path}")

            filename = os.path.basename(resolved_path)
            return resolved_path, filename

    def _resolve_file_path(self, path: str) -> str:
        """Resolve file path (handle relative paths)."""
        # Handle relative paths starting with ./ComfyUI/
        if path.startswith('./ComfyUI/'):
            # Remove ./ComfyUI/ prefix and resolve from ComfyUI root
            relative_path = path[10:]  # Remove './ComfyUI/'
            output_dir = folder_paths.get_output_directory()
            comfyui_root = os.path.dirname(output_dir)
            return os.path.join(comfyui_root, relative_path)

        # Handle other relative paths (relative to input directory)
        elif not os.path.isabs(path):
            input_dir = folder_paths.get_input_directory()
            return os.path.join(input_dir, path)

        # Absolute path
        return path

    def _load_image_from_path(self, file_path: str):
        """Load image from file path using ComfyUI's standard approach."""
        try:
            # Try using the parent class's load_image method
            return super().load_image(file_path)
        except (IndexError, Exception):
            # Fallback to manual loading
            img_pil = Image.open(file_path)
            img_pil = ImageOps.exif_transpose(img_pil).convert("RGBA")

            # Convert to numpy and normalize to [0,1]
            arr = np.array(img_pil).astype(np.float32) / 255.0  # shape (H, W, 4)

            # Split RGB vs. alpha
            rgb = arr[..., :3]      # [H, W, 3]
            alpha = arr[..., 3]     # [H, W]

            # Build image tensor [1, H, W, C]
            image_tensor = torch.from_numpy(rgb).unsqueeze(0)

            # Build mask tensor [1, H, W, 1] with inverted alpha
            mask_tensor = (1.0 - torch.from_numpy(alpha)).unsqueeze(0).unsqueeze(-1)

            return image_tensor, mask_tensor

    def _process_filename_for_file_picker(self, file_path: str, filename_with_ext: bool) -> str:
        """Process filename for file picker mode to return relative path."""
        input_dir = folder_paths.get_input_directory()
        # Normalize path separators for comparison
        normalized_file_path = file_path.replace("\\", "/")
        normalized_input_dir = input_dir.replace("\\", "/")

        # Create relative path starting with "./ComfyUI"
        if normalized_file_path.startswith(normalized_input_dir):
            relative_path = normalized_file_path[len(normalized_input_dir):].lstrip("/")
            full_relative_path = f"./ComfyUI/input/{relative_path}"
        else:
            # If it's not in input directory, just use the basename with ComfyUI prefix
            full_relative_path = f"./ComfyUI/{os.path.basename(file_path)}"

        # If the path contains slashes, return only the actual filename
        if "/" in full_relative_path or "\\" in full_relative_path:
            actual_filename = os.path.basename(full_relative_path)
        else:
            actual_filename = full_relative_path

        return actual_filename
