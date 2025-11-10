#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#

import os
import re
import folder_paths
import urllib.request
import urllib.parse
import json
import pickle
import safetensors.torch
from typing import List
from PIL import Image
import torch
import numpy as np

from ...utils.constants import get_category
from ...utils.log import log
from ...utils.helper import AnyType

any_type = AnyType("*")


class IsFileExists_v1:

    NAME = "Is File Exists"
    SHORTCUT = "f"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "The filename to search for (can be regex pattern if regex mode is enabled)"
                }),
                "search_mode": (["static", "regex"], {
                    "default": "static",
                    "tooltip": "Choose whether to treat filename as static string or regex pattern"
                }),
                "directory": (["output", "input", "temp", "all"], {
                    "default": "output",
                    "tooltip": "Select which directories to search in"
                }),
                "return_type": (["single", "list"], {
                    "default": "single",
                    "tooltip": "Return single result (first found) or list of all matches"
                }),
            },
            "optional": {
                "recursive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Search recursively in subdirectories"
                }),
                "extension": ("STRING", {
                    "default": "",
                    "tooltip": "File extension to match (must include dot: .png, .txt, .latent). If empty, ignores extensions"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("result", "paths", "count", "exists")
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")

    def fn(self, filename: str, search_mode: str = "static", directory: str = "all",
           return_type: str = "single", recursive: bool = True, extension: str = ""):

        if not filename.strip():
            return ("None", "", 0, False)
        try:

            # If extension is specified, check if file has that extension
            if extension.strip():
                # Extension must start with a dot (e.g., .png, .txt, .latent)
                if not extension.startswith('.'):
                    extension = '.' + extension

            # Handle filename extension based on extension field
            filename_base, filename_ext = os.path.splitext(filename)

            if extension.strip():  # extension field is not empty
                if search_mode == "static":
                    filename = filename_base + extension
                    regex_pattern = f"^{re.escape(filename)}$"
                elif search_mode == "regex":
                    # For regex, escape filename_base to handle special characters, then add extension at end
                    regex_pattern = f".*{re.escape(filename_base)}.*{re.escape(extension)}$"
            else:  # extension field is empty
                if search_mode == "static":
                    filename = filename_base
                    regex_pattern = f"^{re.escape(filename_base)}$"
                elif search_mode == "regex":
                    # For regex without extension, escape filename_base to handle special characters
                    regex_pattern = f".*{re.escape(filename_base)}.*"


            # Get directory paths
            search_dirs = self._get_search_directories(directory)

            # Find files
            found_files = self._search_files(filename, regex_pattern, search_dirs, search_mode, recursive, extension)

            # Sort files by creation time (latest first)
            if found_files:
                found_files = self._sort_files_by_creation_time(found_files)
                # Convert to relative paths starting with "./ComfyUI/"
                found_files = self._convert_to_relative_paths(found_files)

            # Process results based on return type
            if return_type == "single":
                if found_files:
                    result = found_files[0]  # First item is now the latest created
                    paths = found_files[0]
                    count = 1
                    found = True
                else:
                    result = "None"
                    paths = ""
                    count = 0
                    found = False
            else:  # return_type == "list"
                if found_files:
                    result = str(found_files)  # List is now sorted by creation time (latest first)
                    paths = "\n".join(found_files)
                    count = len(found_files)
                    found = True
                else:
                    result = "None"
                    paths = ""
                    count = 0
                    found = False

            return (result, paths, count, found)

        except Exception as e:
            log(f"Error in IsFileExists: {str(e)}", None, None, "error")
            return ("None", f"Error: {str(e)}", 0, False)

    def _get_search_directories(self, directory: str) -> List[str]:
        """Get the list of directories to search based on user selection."""
        dirs = []

        if directory == "all":
            dirs.extend([
                folder_paths.get_output_directory(),
                folder_paths.get_input_directory(),
                folder_paths.get_temp_directory()
            ])
        elif directory == "output":
            dirs.append(folder_paths.get_output_directory())
        elif directory == "input":
            dirs.append(folder_paths.get_input_directory())
        elif directory == "temp":
            dirs.append(folder_paths.get_temp_directory())

        # Filter out directories that don't exist
        existing_dirs = []
        for d in dirs:
            if os.path.exists(d):
                existing_dirs.append(d)
            else:
                log(f"Directory does not exist: {d}", None, None, "warning")

        return existing_dirs

    def _search_files(self, filename: str, regex_pattern: str, search_dirs: List[str],
                      search_mode: str, recursive: bool, extension: str) -> List[str]:
        """Search for files in the specified directories."""
        found_files = []

        for search_dir in search_dirs:
            if recursive:
                # Search recursively
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if self._matches_pattern(file, filename, regex_pattern, search_mode, extension):
                            full_path = os.path.join(root, file)
                            found_files.append(full_path)
            else:
                # Search only in the root directory
                try:
                    files = os.listdir(search_dir)
                    for file in files:
                        file_path = os.path.join(search_dir, file)
                        if os.path.isfile(file_path):
                            if self._matches_pattern(file, filename, regex_pattern, search_mode, extension):
                                found_files.append(file_path)
                except OSError as e:
                    log(f"Error accessing directory {search_dir}: {str(e)}", None, None, "error")

        return found_files

    def _sort_files_by_creation_time(self, files: List[str]) -> List[str]:
        """Sort files by creation time, latest first."""
        try:
            def get_creation_time(file_path):
                try:
                    return os.path.getctime(file_path)
                except (OSError, FileNotFoundError):
                    # If we can't get creation time, return 0 (oldest possible time)
                    return 0

            # Sort by creation time in descending order (latest first)
            return sorted(files, key=get_creation_time, reverse=True)
        except Exception as e:
            log(f"Error sorting files by creation time: {str(e)}", None, None, "error")
            return files  # Return original list if sorting fails

    def _convert_to_relative_paths(self, files: List[str]) -> List[str]:
        """Convert absolute file paths to relative paths starting with './ComfyUI/'."""
        relative_files = []

        # Get ComfyUI base directory (parent of current working directory structure)
        output_dir = folder_paths.get_output_directory()

        # Find the ComfyUI root directory (common parent)
        comfyui_root = os.path.dirname(output_dir)  # Assuming output is in ComfyUI/output

        for file_path in files:
            try:
                # Normalize path separators
                normalized_file_path = file_path.replace("\\", "/")
                normalized_comfyui_root = comfyui_root.replace("\\", "/")

                # Create relative path from ComfyUI root
                if normalized_file_path.startswith(normalized_comfyui_root):
                    relative_path = normalized_file_path[len(normalized_comfyui_root):].lstrip("/")
                    relative_files.append(f"./ComfyUI/{relative_path}")
                else:
                    # If file is not under ComfyUI root, just use the filename
                    filename = os.path.basename(file_path)
                    relative_files.append(f"./ComfyUI/{filename}")
            except Exception as e:
                log(f"Error converting path to relative: {file_path} - {str(e)}", None, None, "error")
                # Fallback to just the filename
                filename = os.path.basename(file_path)
                relative_files.append(f"./ComfyUI/{filename}")

        return relative_files

    def _matches_pattern(self, file: str, compare_filename: str, compare_pattern: str, search_mode: str, extension: str) -> bool:
        """Check if a filename matches the given pattern."""

        if search_mode == "static":
            # Case-insensitive static comparison: compare file against compare_filename
            return file.lower() == compare_filename.lower()
        elif search_mode == "regex":
            try:
                # For regex mode: compare file against the pre-built regex pattern (compare_pattern)
                return bool(re.search(compare_pattern, file, re.IGNORECASE))
            except re.error as e:
                log(f"Invalid regex pattern '{compare_pattern}': {str(e)}", None, None, "error")
                return False

        return False

    @classmethod
    def IS_CHANGED(cls, filename, search_mode, directory, return_type, recursive=True, extension=""):
        # This method can be used to determine if the node should be re-executed
        # For file existence checks, we might want to re-execute when inputs change
        return f"{filename}_{search_mode}_{directory}_{return_type}_{recursive}_{extension}"


class LoadFile_v1:

    NAME = "Load File"
    SHORTCUT = "l"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_or_url": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "File path (full or relative) or URL to load"
                }),
                "file_type": (["AUTO", "IMAGE", "MODEL", "VAE", "CLIP", "CONTROLNET", "LATENT", "TEXT", "JSON", "AUDIO", "VIDEO"], {
                    "default": "AUTO",
                    "tooltip": "Type of file to load - determines output format"
                }),
            },
            "optional": {
                "force_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force re-download if URL (ignores cached version)"
                }),
            }
        }

    RETURN_TYPES = (any_type, "STRING")
    RETURN_NAMES = ("output", "path")
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")
    OUTPUT_NODE = False

    @classmethod
    def get_return_types_for_file_type(cls, file_type: str):
        """Get appropriate return types based on file type."""
        if file_type == "IMAGE":
            return ("IMAGE",)  # Return actual image tensor
        elif file_type in ["TEXT", "JSON"]:
            return ("STRING",)  # Return actual text/JSON content as string
        elif file_type == "LATENT":
            return ("LATENT",)  # Return actual latent data
        elif file_type in ["MODEL", "VAE", "CLIP", "CONTROLNET", "AUDIO", "VIDEO"]:
            return ("STRING",)  # Return path for ComfyUI to handle
        else:
            return (any_type,)  # Default fallback to any_type

    def fn(self, path_or_url: str, file_type: str = "AUTO", force_download: bool = False):
        if not path_or_url.strip():
            return (None, "")

        try:
            # Determine if it's a URL or path
            is_url = self._is_url(path_or_url)

            if is_url:
                # Download file to temp directory
                file_path = self._download_file(path_or_url, force_download)
            else:
                # Handle local file path
                file_path = self._resolve_file_path(path_or_url)

            # Verify file exists
            if not os.path.exists(file_path):
                log(f"File not found: {file_path}", None, None, "error")
                return (None, "")

            # Auto-detect file type if needed
            if file_type == "AUTO":
                file_type = self._detect_file_type(file_path)

            # Load file based on type
            result = self._load_file_by_type(file_path, file_type)

            # Generate relative path
            relative_path = self._convert_to_relative_path(file_path)

            return (result, relative_path)

        except Exception as e:
            log(f"Error in LoadFile: {str(e)}", None, None, "error")
            return (None, "")

    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        if not path:
            return False
        return path.startswith(('http://', 'https://'))

    def _download_file(self, url: str, force_download: bool = False) -> str:
        """Download file from URL to temp directory."""
        # Create a safe filename from URL
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        if not filename:
            filename = "downloaded_file"

        # Use temp directory
        temp_dir = folder_paths.get_temp_directory()
        file_path = os.path.join(temp_dir, filename)

        # Check if file already exists and force_download is False
        if os.path.exists(file_path) and not force_download:
            log(f"Using cached file: {file_path}", None, None, "info")
            return file_path

        # Download the file
        log(f"Downloading file from: {url}", None, None, "info")
        urllib.request.urlretrieve(url, file_path)
        log(f"Downloaded to: {file_path}", None, None, "info")

        return file_path

    def _resolve_file_path(self, path: str) -> str:
        """Resolve file path (handle relative paths)."""
        # Handle relative paths starting with ./ComfyUI/
        if path.startswith('./ComfyUI/'):
            # Remove ./ComfyUI/ prefix and resolve from ComfyUI root
            relative_path = path[10:]  # Remove './ComfyUI/'
            output_dir = folder_paths.get_output_directory()
            comfyui_root = os.path.dirname(output_dir)
            return os.path.join(comfyui_root, relative_path)

        # Handle other relative paths
        elif not os.path.isabs(path):
            # Relative to ComfyUI root
            output_dir = folder_paths.get_output_directory()
            comfyui_root = os.path.dirname(output_dir)
            return os.path.join(comfyui_root, path)

        # Absolute path
        return path

    def _detect_file_type(self, file_path: str) -> str:
        """Auto-detect file type based on extension."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']:
            return "IMAGE"
        elif ext in ['.ckpt', '.pt', '.pth', '.bin', '.safetensors']:
            # Try to detect more specific model types by filename patterns
            filename = os.path.basename(file_path).lower()
            if any(word in filename for word in ['vae', 'encoder', 'decoder']):
                return "VAE"
            elif any(word in filename for word in ['clip', 'text_encoder']):
                return "CLIP"
            elif any(word in filename for word in ['controlnet', 'control']):
                return "CONTROLNET"
            else:
                return "MODEL"
        elif ext in ['.txt', '.md', '.log', '.csv']:
            return "TEXT"
        elif ext in ['.json', '.jsonl']:
            return "JSON"
        elif ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            return "AUDIO"
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return "VIDEO"
        elif ext in ['.latent', '.tensor']:
            return "LATENT"
        else:
            return "TEXT"  # Default fallback

    def _load_file_by_type(self, file_path: str, file_type: str):
        """Load file based on specified type."""
        if file_type == "IMAGE":
            return self._load_image(file_path)
        elif file_type == "TEXT":
            return self._load_text(file_path)
        elif file_type == "JSON":
            return self._load_json(file_path)
        elif file_type == "LATENT":
            return self._load_latent(file_path)
        elif file_type in ["MODEL", "VAE", "CLIP", "CONTROLNET", "AUDIO", "VIDEO"]:
            # For audio/video, return path as ComfyUI handles these
            return self._convert_to_relative_path(file_path)
        else:
            # Default: try to load as text
            try:
                return self._load_text(file_path)
            except:
                # Fallback to returning path if can't load as text
                return self._convert_to_relative_path(file_path)

    def _load_image(self, file_path: str):
        """Load image file."""
        img = Image.open(file_path)
        img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        # Ensure proper format: [batch, height, width, channels]
        if len(img_array.shape) == 3:  # [height, width, channels]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension
        else:
            img_tensor = torch.from_numpy(img_array)
        return img_tensor

    def _load_text(self, file_path: str) -> str:
        """Load text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_json(self, file_path: str):
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            # Load JSON and return as formatted string for ComfyUI compatibility
            json_data = json.load(f)
            return json.dumps(json_data, indent=2, ensure_ascii=False)

    def _load_model(self, file_path: str) -> str:
        """Load model file - return path for ComfyUI to handle."""
        # For models, we typically return the path for ComfyUI to load
        return file_path

    def _load_latent(self, file_path: str):
        """Load latent file using ComfyUI's standard approach."""
        try:
            # Use safetensors as ComfyUI's LoadLatent does
            latent_data = safetensors.torch.load_file(file_path, device="cpu")
            multiplier = 1.0
            if "latent_format_version_0" not in latent_data:
                multiplier = 1.0 / 0.18215
            samples = {"samples": latent_data["latent_tensor"].float() * multiplier}
            return samples
        except Exception as e:
            log(f"Error loading latent file with safetensors: {file_path} - {str(e)}", None, None, "error")
            # Fallback to torch.load for older formats
            try:
                latent_data = torch.load(file_path, map_location='cpu', weights_only=False)
                # If it's already in the correct format, return it
                if isinstance(latent_data, dict) and "samples" in latent_data:
                    return latent_data
                # If it's a raw tensor, wrap it in the expected format
                elif isinstance(latent_data, torch.Tensor):
                    return {"samples": latent_data}
                else:
                    # Unknown format, try to convert
                    log(f"Unknown latent format in file: {file_path}", None, None, "warning")
                    return {"samples": latent_data}
            except (pickle.PickleError, RuntimeError, ValueError, KeyError) as e2:
                if "invalid load key" in str(e2) or "pickle" in str(e2).lower():
                    log(f"Latent file appears corrupted or uses unsupported format: {file_path} - {str(e2)}", None, None, "error")
                else:
                    log(f"Error loading latent file {file_path}: {str(e2)}", None, None, "error")
                # Fallback to returning the path
                return file_path
            except Exception as e2:
                log(f"Unexpected error loading latent file {file_path}: {str(e2)}", None, None, "error")
                # Fallback to returning the path
                return file_path

    def _convert_to_relative_path(self, file_path: str) -> str:
        """Convert absolute file path to relative path starting with './ComfyUI/'."""
        try:
            # Normalize path separators
            normalized_file_path = file_path.replace("\\", "/")

            # Get ComfyUI base directory (parent of current working directory structure)
            output_dir = folder_paths.get_output_directory()
            comfyui_root = os.path.dirname(output_dir)  # Assuming output is in ComfyUI/output
            normalized_comfyui_root = comfyui_root.replace("\\", "/")

            # Create relative path from ComfyUI root
            if normalized_file_path.startswith(normalized_comfyui_root):
                relative_path = normalized_file_path[len(normalized_comfyui_root):].lstrip("/")
                return f"./ComfyUI/{relative_path}"
            else:
                # If file is not under ComfyUI root, just use the filename
                filename = os.path.basename(file_path)
                return f"./ComfyUI/{filename}"
        except Exception as e:
            log(f"Error converting path to relative: {file_path} - {str(e)}", None, None, "error")
            # Fallback to just the filename
            filename = os.path.basename(file_path)
            return f"./ComfyUI/{filename}"

    @classmethod
    def IS_CHANGED(cls, path_or_url, file_type="AUTO", force_download=False):
        # Handle None or empty path_or_url
        if not path_or_url:
            return ""

        # For URLs, always check if force_download is True
        if path_or_url.startswith(('http://', 'https://')) and force_download:
            import time
            return str(time.time())
        return path_or_url


class SaveFile_v1:

    NAME = "Save File"
    SHORTCUT = "s"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content": (any_type, {
                    "tooltip": "The content to save - can be IMAGE, LATENT, STRING, JSON, or tensor data"
                }),
                "path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "File path (URL, full path, or relative path). Smart path handling based on 'ComfyUI/' presence"
                }),
                "file_type": (["AUTO", "IMAGE", "MODEL", "VAE", "CLIP", "CONTROLNET", "LATENT", "TEXT", "JSON", "AUDIO", "VIDEO"], {
                    "default": "AUTO",
                    "tooltip": "Type of file to save - determines save format"
                }),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow overwriting existing files"
                }),
                "create_directories": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Create parent directories if they don't exist"
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("saved", "reason", "final_path")
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")
    OUTPUT_NODE = True

    def fn(self, content, path: str, file_type: str = "AUTO", overwrite: bool = False, create_directories: bool = True):
        if not path.strip():
            return (False, "Empty path provided", "")

        try:
            # Process the path according to the rules
            final_path = self._process_save_path(path.strip())

            # Auto-detect file type if needed
            if file_type == "AUTO":
                file_type = self._detect_content_type(content, final_path)
                if not file_type:
                    return (False, "Could not detect file type from content", final_path)

            # Check if file exists and handle overwrite
            if os.path.exists(final_path) and not overwrite:
                return (False, f"File already exists and overwrite is disabled: {final_path}", final_path)

            # Create parent directories if needed
            if create_directories:
                parent_dir = os.path.dirname(final_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                    log(f"Created directories: {parent_dir}", None, None, "info")

            # Save based on detected/specified type
            success, reason = self._save_content_by_type(content, final_path, file_type)

            if success:
                # Convert to relative path for return
                relative_path = self._convert_to_relative_path(final_path)
                log(f"Successfully saved file: {relative_path}", None, None, "info")
                return (True, f"File saved successfully as {file_type}", relative_path)
            else:
                return (False, reason, final_path)

        except Exception as e:
            error_msg = f"Error saving file: {str(e)}"
            log(error_msg, None, None, "error")
            return (False, error_msg, path)

    def _process_save_path(self, path: str) -> str:
        """Process save path according to the rules."""
        # Get ComfyUI root directory
        output_dir = folder_paths.get_output_directory()
        comfyui_root = os.path.dirname(output_dir)

        # Handle URLs - download first
        if path.startswith(('http://', 'https://')):
            filename = os.path.basename(urllib.parse.urlparse(path).path)
            if not filename:
                filename = "downloaded_file"
            return os.path.join(folder_paths.get_output_directory(), filename)

        # Check if path contains "ComfyUI/"
        if "ComfyUI/" in path:
            # Extract the part after "ComfyUI/"
            comfyui_index = path.rfind("ComfyUI/")
            relative_part = path[comfyui_index + 8:]  # Skip "ComfyUI/"
            return os.path.join(comfyui_root, relative_part)

        # Handle relative paths starting with ./ComfyUI/
        elif path.startswith('./ComfyUI/'):
            relative_part = path[10:]  # Remove './ComfyUI/'
            return os.path.join(comfyui_root, relative_part)

        # Handle absolute paths
        elif os.path.isabs(path):
            return path

        # Default: save in output directory
        else:
            return os.path.join(folder_paths.get_output_directory(), os.path.basename(path))

    def _detect_content_type(self, content, file_path: str) -> str:
        """Detect content type from the actual content and file extension."""
        # First try to detect from content type
        if isinstance(content, torch.Tensor):
            # Check tensor dimensions to guess type
            if len(content.shape) == 4 and content.shape[-1] == 3:  # [B, H, W, C] image
                return "IMAGE"
            elif len(content.shape) == 4 and content.shape[1] == 4:  # [B, C, H, W] latent
                return "LATENT"
            else:
                return "LATENT"  # Default for unknown tensors

        elif isinstance(content, dict):
            if "samples" in content:
                return "LATENT"
            else:
                return "JSON"

        elif isinstance(content, (list, tuple)):
            return "JSON"

        elif isinstance(content, str):
            return "TEXT"

        # Fallback to file extension detection
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']:
            return "IMAGE"
        elif ext in ['.txt', '.md', '.log', '.csv']:
            return "TEXT"
        elif ext in ['.json', '.jsonl']:
            return "JSON"
        elif ext in ['.latent', '.tensor']:
            return "LATENT"
        elif ext in ['.ckpt', '.pt', '.pth', '.bin', '.safetensors']:
            return "MODEL"

        # Could not detect
        return None

    def _save_content_by_type(self, content, file_path: str, file_type: str) -> tuple:
        """Save content based on type. Returns (success, reason)."""
        try:
            if file_type == "IMAGE":
                return self._save_image(content, file_path)
            elif file_type == "TEXT":
                return self._save_text(content, file_path)
            elif file_type == "JSON":
                return self._save_json(content, file_path)
            elif file_type == "LATENT":
                return self._save_latent(content, file_path)
            elif file_type in ["MODEL", "VAE", "CLIP", "CONTROLNET"]:
                return self._save_model(content, file_path)
            else:
                return (False, f"Unsupported file type: {file_type}")

        except Exception as e:
            return (False, f"Error saving {file_type}: {str(e)}")

    def _save_image(self, content, file_path: str) -> tuple:
        """Save image content."""
        try:
            if isinstance(content, torch.Tensor):
                # Convert tensor to PIL Image
                if len(content.shape) == 4:  # [B, H, W, C]
                    img_array = content[0].cpu().numpy()  # Take first batch
                elif len(content.shape) == 3:  # [H, W, C]
                    img_array = content.cpu().numpy()
                else:
                    return (False, "Invalid image tensor shape")

                # Convert from [0,1] to [0,255]
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)

                img = Image.fromarray(img_array)
                img.save(file_path)
                return (True, f"Image saved to {file_path}")

            elif hasattr(content, 'save'):  # PIL Image
                content.save(file_path)
                return (True, f"PIL Image saved to {file_path}")

            else:
                return (False, "Content is not a valid image format")

        except Exception as e:
            return (False, f"Error saving image: {str(e)}")

    def _save_text(self, content, file_path: str) -> tuple:
        """Save text content."""
        try:
            text_content = str(content)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            return (True, f"Text saved to {file_path}")
        except Exception as e:
            return (False, f"Error saving text: {str(e)}")

    def _save_json(self, content, file_path: str) -> tuple:
        """Save JSON content."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            return (True, f"JSON saved to {file_path}")
        except Exception as e:
            return (False, f"Error saving JSON: {str(e)}")

    def _save_latent(self, content, file_path: str) -> tuple:
        """Save latent content using ComfyUI's standard format."""
        try:
            if isinstance(content, dict) and "samples" in content:
                # Use ComfyUI's standard latent format
                output = {}
                output["latent_tensor"] = content["samples"].contiguous()
                output["latent_format_version_0"] = torch.tensor([])

                # Save using safetensors (ComfyUI standard)
                import comfy.utils
                comfy.utils.save_torch_file(output, file_path)
                return (True, f"Latent saved to {file_path}")

            elif isinstance(content, torch.Tensor):
                # Wrap tensor in ComfyUI format
                output = {}
                output["latent_tensor"] = content.contiguous()
                output["latent_format_version_0"] = torch.tensor([])

                import comfy.utils
                comfy.utils.save_torch_file(output, file_path)
                return (True, f"Tensor saved as latent to {file_path}")

            else:
                return (False, "Content is not a valid latent format")

        except Exception as e:
            return (False, f"Error saving latent: {str(e)}")

    def _save_model(self, content, file_path: str) -> tuple:
        """Save model content."""
        try:
            if isinstance(content, str):
                # Assume it's a path to copy from
                if os.path.exists(content):
                    import shutil
                    shutil.copy2(content, file_path)
                    return (True, f"Model file copied to {file_path}")
                else:
                    return (False, f"Source model file not found: {content}")

            elif isinstance(content, dict) or hasattr(content, 'state_dict'):
                # Save as torch file
                torch.save(content, file_path)
                return (True, f"Model saved to {file_path}")

            else:
                return (False, "Content is not a valid model format")

        except Exception as e:
            return (False, f"Error saving model: {str(e)}")

    def _convert_to_relative_path(self, file_path: str) -> str:
        """Convert absolute path to relative ComfyUI path."""
        try:
            # Normalize path separators
            normalized_file_path = file_path.replace("\\", "/")

            # Get ComfyUI base directory
            output_dir = folder_paths.get_output_directory()
            comfyui_root = os.path.dirname(output_dir)
            normalized_comfyui_root = comfyui_root.replace("\\", "/")

            # Create relative path from ComfyUI root
            if normalized_file_path.startswith(normalized_comfyui_root):
                relative_path = normalized_file_path[len(normalized_comfyui_root):].lstrip("/")
                return f"./ComfyUI/{relative_path}"
            else:
                # If file is not under ComfyUI root, just use the filename
                filename = os.path.basename(file_path)
                return f"./ComfyUI/{filename}"
        except Exception as e:
            log(f"Error converting path to relative: {file_path} - {str(e)}", None, None, "error")
            return file_path

    @classmethod
    def IS_CHANGED(cls, content, path, file_type="AUTO", overwrite=False, create_directories=True):
        # Always re-execute when inputs change
        return f"{path}_{file_type}_{overwrite}_{create_directories}"

