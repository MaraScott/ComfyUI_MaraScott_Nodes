"""
SeedAtlas_v1 custom node for ComfyUI.

Generates an atlas (grid) of images across a deterministic range of seeds and
returns the stitched grid image, CSV metadata text, JSON metadata text, and
optionally the list of individual images. Optionally saves outputs to disk.

Targets: ComfyUI >= 0.3.29, Python 3.12, Torch >= 2.6.0.

Example usage:
1) Text2Img atlas across 24 seeds, 4x6 grid.
2) Image2Image with provided latent and denoise=0.55.
3) Character/scene formula with character_id=7, scene_id=3, stride=2.
"""

import os
import io
import ast
import csv
import json
import math
import time
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.samplers
from nodes import CLIPTextEncode, VAEDecode, common_ksampler
from ...utils.constants import get_name, get_category


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _validate_multiple_of_8(x: int, name: str) -> None:
    if x % 8 != 0:
        raise ValueError(f"{name} must be a multiple of 8 (got {x}).")


def _to_pil(img_tensor: torch.Tensor) -> Image.Image:
    # img_tensor: [H,W,C] or [1,H,W,C] with range [0,1]
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = (img_tensor.clamp(0, 1).cpu().numpy() * 255.0).astype("uint8")
    return Image.fromarray(img)


def _from_pil(img: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(torch.ByteTensor(bytearray(img.tobytes())).numpy())
    # Convert bytes back to HWC
    w, h = img.size
    c = len(img.getbands())
    arr = arr.view(h, w, c).to(torch.float32) / 255.0
    return arr.unsqueeze(0)  # [1,H,W,C]


def _safe_eval_seed_formula(expr: str, local_vars: Dict[str, int]) -> int:
    # Parse a restricted arithmetic expression safely via AST
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub, ast.Name, ast.Load,
        ast.Constant
    )

    def _check(node: ast.AST):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Illegal expression element: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            _check(child)

    try:
        tree = ast.parse(expr, mode="eval")
        _check(tree)
        code = compile(tree, "<seed_formula>", "eval")
        val = eval(code, {"__builtins__": {}}, local_vars)
    except Exception as e:
        # Fallback to default formula
        val = local_vars["start_seed"] + local_vars["i"] * local_vars["stride"] \
              + local_vars["character_id"] * 100 + local_vars["scene_id"] * 10
    try:
        return int(val)
    except Exception:
        return local_vars["start_seed"]


def _make_empty_latent(width: int, height: int, batch: int = 1) -> Dict[str, torch.Tensor]:
    h = height // 8
    w = width // 8
    samples = torch.zeros((batch, 4, h, w), dtype=torch.float32)
    return {"samples": samples}


def _draw_label(img: Image.Image, text: str, font_size: int) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw = ImageDraw.Draw(img)
    margin = 2
    if font is not None:
        draw.text((margin, img.height - font_size - margin), text, fill=(255, 255, 255), font=font)
    else:
        draw.text((margin, img.height - font_size - margin), text, fill=(255, 255, 255))


def _stitch_grid(images: List[Image.Image], rows: int, cols: int, padding: int, order: str,
                 fill_label: bool, font_size: int, labels: List[str]) -> Image.Image:
    if not images:
        raise ValueError("No images to stitch.")
    w, h = images[0].size
    canvas_w = cols * w + (cols - 1) * padding
    canvas_h = rows * h + (rows - 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))

    def idx(r: int, c: int) -> int:
        return r * cols + c if order == "row-major" else c * rows + r

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            x = c * (w + padding)
            y = r * (h + padding)
            if i < len(images):
                img = images[i]
            else:
                img = Image.new("RGB", (w, h), (0, 0, 0))
                if fill_label and labels and i < len(labels):
                    _draw_label(img, labels[i], font_size)
            canvas.paste(img, (x, y))
    return canvas


class SeedAtlas_v1:
    NAME = "Seed Atlas"
    SHORTCUT = "i"

    @classmethod
    def INPUT_TYPES(cls):
        sampler_names = tuple(x.name for x in comfy.samplers.KSampler.SAMPLERS)
        scheduler_names = tuple(x.name for x in comfy.samplers.KSampler.SCHEDULERS)
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "sampler_name": ("STRING", {"default": sampler_names[0] if sampler_names else "euler", "choices": sampler_names}),
                "scheduler": ("STRING", {"default": scheduler_names[0] if scheduler_names else "normal", "choices": scheduler_names}),
                "prompt": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.05}),
                "start_seed": ("INT", {"default": 10000}),
                "count": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
                "rows": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "cols": ("INT", {"default": 6, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "base_seed": ("INT", {"default": 12000}),
                "character_id": ("INT", {"default": 0}),
                "scene_id": ("INT", {"default": 0}),
                "seed_formula": ("STRING", {"default": "seed = start_seed + i*stride + character_id*100 + scene_id*10"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent": ("LATENT", {}),
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "save_to_disk": ("BOOLEAN", {"default": True}),
                "save_dir": ("STRING", {"default": "output/seed_atlas"}),
                "filename_prefix": ("STRING", {"default": "Atlas"}),
                "return_individuals": ("BOOLEAN", {"default": False}),
                "draw_labels": ("BOOLEAN", {"default": True}),
                "font_size": ("INT", {"default": 14, "min": 6, "max": 32, "step": 1}),
                "grid_padding": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "tile_order": ("STRING", {"default": "row-major", "choices": ("row-major", "col-major")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")

    def _encode_text(self, clip, text: str):
        # Reuse ComfyUI encoding behavior
        return CLIPTextEncode().encode(clip, text)[0]

    def _decode(self, vae, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        return VAEDecode().decode(vae, samples)[0]

    def fn(
        self,
        model, clip, vae,
        sampler_name: str, scheduler: str,
        prompt: str, negative: str,
        width: int, height: int,
        steps: int, cfg: float,
        start_seed: int, count: int, stride: int, rows: int, cols: int,
        base_seed: int = 12000, character_id: int = 0, scene_id: int = 0,
        seed_formula: str = "seed = start_seed + i*stride + character_id*100 + scene_id*10",
        denoise: float = 1.0, latent: Dict[str, torch.Tensor] | None = None,
        clip_skip: int = 1, batch_size: int = 1,
        save_to_disk: bool = True, save_dir: str = "output/seed_atlas", filename_prefix: str = "Atlas",
        return_individuals: bool = False, draw_labels: bool = True, font_size: int = 14,
        grid_padding: int = 2, tile_order: str = "row-major",
    ):
        _validate_multiple_of_8(width, "width")
        _validate_multiple_of_8(height, "height")

        max_cells = rows * cols
        truncated = False
        if count > max_cells:
            truncated = True
            count = max_cells
        if max_cells < count:
            raise ValueError("rows * cols must be >= count")

        # Encoding
        positive = self._encode_text(clip, prompt)
        negative_c = self._encode_text(clip, negative if negative is not None else "")

        # Optionally attempt clip_skip
        try:
            if hasattr(clip, "set_clip_skip") and isinstance(clip_skip, int) and clip_skip > 1:
                clip.set_clip_skip(clip_skip)
        except Exception:
            pass

        # Prepare seeds
        seeds: List[int] = []
        for i in range(count):
            local_ns = {
                "i": i, "start_seed": start_seed, "stride": stride, "base_seed": base_seed,
                "character_id": character_id, "scene_id": scene_id,
            }
            seed_val = _safe_eval_seed_formula(seed_formula.split("=", 1)[-1].strip(), local_ns)
            seeds.append(seed_val)

        # Generation loop
        images_pil: List[Image.Image] = []
        labels: List[str] = []
        metadata_rows: List[Dict[str, Any]] = []

        t0 = time.strftime("%Y%m%d_%H%M%S")

        # Create empty latent if not provided
        base_latent = latent if latent is not None else _make_empty_latent(width, height, batch=1)

        for idx, seed in enumerate(seeds):
            print(f"[SeedAtlas_v1] generating {idx+1}/{count} seed={seed}")
            # Make a copy of base latent for this item
            item_latent = {"samples": base_latent["samples"].clone()}

            # Sample
            out_latent = common_ksampler(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative_c,
                latent=item_latent,
                denoise=denoise,
            )

            # Decode to image tensor [B,H,W,C]
            img_tensor = self._decode(vae, out_latent)
            # Convert first frame to PIL and free VRAM
            pil_img = _to_pil(img_tensor)
            if draw_labels:
                _draw_label(pil_img, f"seed={seed}", font_size)
            images_pil.append(pil_img)
            labels.append(f"seed={seed}")

            # Metadata
            r = idx // cols if tile_order == "row-major" else idx % rows
            c = idx % cols if tile_order == "row-major" else idx // rows
            row = {
                "seed": seed,
                "row": r,
                "col": c,
                "prompt": prompt,
                "negative": negative,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "sampler": sampler_name,
                "scheduler": scheduler,
                "model_hash": getattr(model, "model_hash", "unknown"),
                "clip_hash": getattr(clip, "model_hash", "unknown"),
                "vae_hash": getattr(vae, "model_hash", "unknown"),
                "timestamp": t0,
            }
            if truncated:
                row["truncated"] = True
            metadata_rows.append(row)

        # Pad remaining cells (visual only)
        if len(images_pil) < rows * cols:
            pad = rows * cols - len(images_pil)
            for _ in range(pad):
                img = Image.new("RGB", (width, height), (0, 0, 0))
                if draw_labels:
                    _draw_label(img, "—", font_size)
                images_pil.append(img)
                labels.append("—")

        # Stitch grid
        grid_img = _stitch_grid(images_pil[: rows * cols], rows, cols, grid_padding, tile_order, draw_labels, font_size, labels)

        # Prepare outputs
        # CSV
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else [])
        if metadata_rows:
            writer.writeheader()
            for row in metadata_rows:
                writer.writerow(row)
        csv_text = csv_buf.getvalue()
        # JSON
        json_text = json.dumps(metadata_rows, ensure_ascii=False, indent=2)

        # Save to disk
        if save_to_disk:
            os.makedirs(save_dir, exist_ok=True)
            cells_dir = os.path.join(save_dir, "cells")
            os.makedirs(cells_dir, exist_ok=True)
            startseed = seeds[0] if seeds else start_seed
            endseed = seeds[-1] if seeds else start_seed
            grid_name = f"{filename_prefix}_mh-{row.get('model_hash','unknown')}_{startseed}-{endseed}_{rows}x{cols}_{width}x{height}_{steps}st_{cfg}cfg.png"
            grid_path = os.path.join(save_dir, grid_name)
            grid_img.save(grid_path, format="PNG")
            # Save individuals if requested
            if return_individuals:
                for i, (pil_img, rowm) in enumerate(zip(images_pil, metadata_rows)):
                    r = rowm["row"]; c = rowm["col"]; s = rowm["seed"]
                    pil_img.save(os.path.join(cells_dir, f"{filename_prefix}_r{r}_c{c}_seed{s}.png"), format="PNG")
            # Save CSV/JSON
            with open(os.path.join(save_dir, f"{filename_prefix}.csv"), "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)
            with open(os.path.join(save_dir, f"{filename_prefix}.json"), "w", encoding="utf-8") as f:
                f.write(json_text)

        # Return ComfyUI image tensor for grid
        grid_tensor = _from_pil(grid_img)

        if return_individuals:
            # Convert individuals back to tensors list
            indiv_tensors = [ _from_pil(p) for p in images_pil[:len(metadata_rows)] ]
            return (grid_tensor, csv_text, json_text, indiv_tensors)
        else:
            return (grid_tensor, csv_text, json_text)
