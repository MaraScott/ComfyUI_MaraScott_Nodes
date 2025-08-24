import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
import json
import math
import time
import statistics as stats

from transformers import (
    AutoImageProcessor,
    ViTImageProcessor,
    AutoModelForImageClassification,
    CLIPProcessor,
    CLIPModel,
)
from ...utils.constants import get_name, get_category

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# -------------------------
# Repositories (HF)
# -------------------------
REPO_NSFW = "LukeJacob2023/nsfw-image-detector"           # drawings, hentai, neutral, porn, sexy
REPO_VIOLENCE = "jaranohaal/vit-base-violence-detection"  # violent vs non-violent
REPO_CLIP = "openai/clip-vit-base-patch32"               # zero-shot horror/gore via text prompts - openai/clip-vit-base-patch32 | openai/clip-vit-base-patch16 | openai/clip-vit-large-patch14

# -------------------------
# Age profile presets (PG 0-12)
# -------------------------
PROFILES = {
    "Strict (0-6)": {
        "nsfw_sum": 0.99,
        "nsfw_peak": 0.995,
        "violence": 0.75,
        "horror": 0.995,
        "weap": 0.96,
        "blood": 0.98
    },
    "Moderate (7-9)":     {
        "nsfw_sum": 0.99,
        "nsfw_peak": 0.995,
        "violence": 0.71,
        "horror": 0.995,
        "weap": 0.96,
        "blood": 0.98
    },
    "Older kids (10-12)": {
        "nsfw_sum": 0.99,
        "nsfw_peak": 0.995,
        "violence": 0.72,
        "horror": 0.995,
        "weap": 0.96,
        "blood": 0.98
    },
}

# Lazy, global singletons (loaded only when needed)
_NSFW_MODEL = None
_NSFW_PROC = None
_NSFW_ID2LABEL = None

_VIO_MODEL = None
_VIO_PROC = None
_VIO_ID2LABEL = None

_CLIP_MODEL = None
_CLIP_PROC = None
_CLIP_TXT_CACHE = {}

# Horror/gore prompts (conservative for PG 0-12)
# POSITIVE (horror/gore)
_HORROR_TEXTS = [
    "graphic gore with blood",
    "severed limb or exposed organs",
    "corpse or dead body",
    "graphic injury with blood splatter",
    "horror scene with terrifying imagery",
]

_WEAPON_TXT = [
    "handgun pistol weapon",
    "rifle gun weapon",
    "knife blade weapon",
    "sword blade weapon",
    "explosive grenade"
]
_BLOOD_TXT = [
    "visible blood",
    "bloody wound",
    "bloodstain on clothes",
    "blood splatter"
]
_NEG_TXT = [
    "toy water gun",
    "kitchen tool not weapon",
    "ketchup stain",
    "red paint",
    "lipstick on lips",
    "cosmetic blush makeup",
    "red face paint",
    "kids face paint",
    "stage makeup without blood",
    "costume party makeup",
    "clown makeup without blood"
]

_COSMETICS_TXT = [
    "lipstick on lips",
    "red face paint",
    "kids face paint",
    "stage makeup without blood",
    "clown makeup without blood",
    "cosmetic blush makeup"
]

_SEX_POS_TXT = [
    "exposed cleavage", "lingerie underwear", "bikini swimsuit cleavage",
    "nude breasts", "topless woman", "erotic suggestive pose",
    "buttocks exposed", "undressing in bedroom"
]
_SEX_NEG_TXT = [
    "close-up face only", "forehead hairline", "scalp close-up",
    "regular portrait", "fully clothed person", "family photo",
    "workplace headshot", "school photo"
]

# NEGATIVE / SAFE contexts (more real-world portrait options)
_SAFE_TEXTS = [
    "kid-friendly cartoon",
    "everyday family photo",
    "nature landscape",
    "studio portrait photo",
    "fashion editorial photo",
    "athlete or dancer in motion",
    "child smiling portrait",
    "kid with face paint",
    "costume party without blood",
    "halloween face paint (no gore)",
    "cartoon cat makeup",
    "school event photo",
    "family photo",
    "school event photo"
]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -------------------------
# Model loaders (conditional)
# -------------------------
def _load_nsfw():
    global _NSFW_MODEL, _NSFW_PROC, _NSFW_ID2LABEL
    if _NSFW_MODEL is None:
        _NSFW_PROC = AutoImageProcessor.from_pretrained(REPO_NSFW, use_fast=True)
        _NSFW_MODEL = AutoModelForImageClassification.from_pretrained(REPO_NSFW)
        _NSFW_MODEL.to(_DEVICE)        # do for _NSFW_MODEL and _VIO_MODEL and _CLIP_MODEL
        _NSFW_MODEL.eval()
        # Optional if CUDA: half precision for vision models
        if _DEVICE.type == "cuda":
            try: _NSFW_MODEL.half()
            except Exception: pass
        _NSFW_ID2LABEL = _NSFW_MODEL.config.id2label

def _load_violence():
    global _VIO_MODEL, _VIO_PROC, _VIO_ID2LABEL
    if _VIO_MODEL is None:
        _VIO_PROC = ViTImageProcessor.from_pretrained(REPO_VIOLENCE)
        _VIO_MODEL = AutoModelForImageClassification.from_pretrained(REPO_VIOLENCE)
        _VIO_MODEL.to(_DEVICE)        # do for _NSFW_MODEL and _VIO_MODEL and _CLIP_MODEL
        _VIO_MODEL.eval()
        # Optional if CUDA: half precision for vision models
        if _DEVICE.type == "cuda":
            try: _VIO_MODEL.half()
            except Exception: pass
        _VIO_ID2LABEL = _VIO_MODEL.config.id2label

def _load_clip():
    global _CLIP_MODEL, _CLIP_PROC
    if _CLIP_MODEL is None:
        _CLIP_PROC = CLIPProcessor.from_pretrained(REPO_CLIP)
        _CLIP_MODEL = CLIPModel.from_pretrained(REPO_CLIP)
        _CLIP_MODEL.to(_DEVICE)        # do for _NSFW_MODEL and _VIO_MODEL and _CLIP_MODEL
        _CLIP_MODEL.eval()
        # Optional if CUDA: half precision for vision models
        if _DEVICE.type == "cuda":
            try: _CLIP_MODEL.half()
            except Exception: pass

# -------------------------
# Utils
# -------------------------
def _tensor_to_pil_batch(image: torch.Tensor) -> List[Image.Image]:
    """
    ComfyUI IMAGE is float tensor in [0,1], accepts [B,H,W,C], [B,C,H,W], or [H,W,C].
    """
    t = image
    if t.dim() == 3:
        t = t.unsqueeze(0)
    # to NHWC
    if t.shape[-1] in (1, 3):
        nhwc = t
    else:
        nhwc = t.permute(0, 2, 3, 1)

    imgs = []
    for i in range(nhwc.shape[0]):
        arr = nhwc[i].detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        imgs.append(Image.fromarray(arr, mode="RGB"))
    return imgs

def _tile_pils(pil_img: Image.Image, tile: int = 384, overlap: int = 64) -> List[Image.Image]:
    """Split into overlapping tiles to catch small unsafe details."""
    w, h = pil_img.size
    out = []
    x = 0
    while x < w:
        y = 0
        x2 = min(x + tile, w)
        while y < h:
            y2 = min(y + tile, h)
            out.append(pil_img.crop((x, y, x2, y2)))
            if y2 == h:
                break
            y += tile - overlap
        if x2 == w:
            break
        x += tile - overlap
    return out

def _pg_decision(
    class_probs: Dict[str, float],
    violence_p: float,
    horror_p: float,
    weapon_p: float,
    blood_p: float,
    cosmetics_p: float,
    sexual_ctx_p: float,
    nsfw_sum_thresh: float,
    nsfw_peak_thresh: float,
    violence_thresh: float,
    horror_thresh: float,
    weapon_thresh: float,
    blood_thresh: float,
    cosmetics_gate: float = 0.65,
    sexual_gate: float = 0.55,
) -> Tuple[int, float, Dict[str, float], str]:
    """
    SAFE = 1, NOT SAFE = 0.
    Sexual NSFW classes considered: porn, sexy, hentai (from the NSFW head).
    Extra safety classes: violence, horror/gore, weapon, blood (from other heads).
    """

    nsfw_classes = ("porn", "sexy", "hentai")
    nsfw_sum  = sum(class_probs.get(k, 0.0) for k in nsfw_classes)
    nsfw_peaks = {k: class_probs.get(k, 0.0) for k in nsfw_classes}
    top_nsfw  = max(nsfw_classes, key=lambda k: nsfw_peaks[k])
    nsfw_peak = nsfw_peaks[top_nsfw]

    flags = []

    # --- NSFW gating ---
    nsfw_should_flag = False
    if nsfw_sum >= nsfw_sum_thresh:
        nsfw_should_flag = True
    elif nsfw_peak >= nsfw_peak_thresh:
        if top_nsfw == "sexy":
            # need sexual context, and don't let plain face/makeup trip it
            nsfw_should_flag = (sexual_ctx_p >= sexual_gate) and (cosmetics_p < cosmetics_gate)
        else:
            # porn/hentai -> no cosmetics escape hatch
            nsfw_should_flag = True

    if nsfw_should_flag:
        flags.append(f"nsfw:{top_nsfw}={fmt_max3(nsfw_peak)}")

    # ---- Violence / Weapons (direct thresholds) ----
    if violence_thresh < 1e8 and violence_p >= violence_thresh:
        flags.append(f"violent={fmt_max3(violence_p)}")
    if weapon_thresh < 1e8 and weapon_p >= weapon_thresh:
        flags.append(f"weapon={fmt_max3(weapon_p)}")

    # ---- Horror needs corroboration (not just spooky makeup) ----
    horror_concord = max(violence_p, weapon_p, blood_p)
    if horror_thresh < 1e8 and horror_p >= horror_thresh and horror_concord >= 0.60:
        flags.append(f"horror/gore={fmt_max3(horror_p)}")

    # ---- Blood needs corroboration and should be low when cosmetics is high ----
    blood_concord = max(violence_p, weapon_p, horror_p)
    if blood_thresh < 1e8 and blood_p >= blood_thresh:
        if blood_concord >= 0.55 and cosmetics_p < cosmetics_gate:
            flags.append(f"blood={fmt_max3(blood_p)}")

    # ---- Final decision ----
    classification = 1 if not flags else 0
    reason = (
        "Content appears kid-friendly: sexual/violent/horror cues are low."
        if classification == 1
        else "Flagged due to " + ", ".join(flags) + "."
    )

    aggregate_risk = max(nsfw_sum, nsfw_peak, violence_p, horror_p, weapon_p, blood_p, cosmetics_p)
    extras = {
        "nsfw_sum": float(nsfw_sum),
        "nsfw_peak": float(nsfw_peak),
        "violence": float(violence_p),
        "horror": float(horror_p),
        "weapon": float(weapon_p),
        "blood": float(blood_p),
        "cosmetics": float(cosmetics_p),
        "sexual_ctx": float(sexual_ctx_p),
    }
    return classification, aggregate_risk, extras, reason

def fmt_max3(x: float) -> str:
    return f"{math.trunc(x*1000)/1000:.3f}"

def fmt_max_dict3(d: Dict[str, float]) -> Dict[str, float]:
    return {k: fmt_max3(v) for k, v in d.items()}

def _proc_image_batch(proc, pil_list: List[Image.Image]):
    inputs = proc(images=pil_list, return_tensors="pt")
    return {k: v.to(_DEVICE) for k, v in inputs.items()}

def _nsfw_probabilities_batch(pils: List[Image.Image]) -> List[Dict[str, float]]:
    _load_nsfw()
    inputs = _proc_image_batch(_NSFW_PROC, pils)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=_DEVICE.type=="cuda"):
        logits = _NSFW_MODEL(**inputs).logits
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return [
        { _NSFW_ID2LABEL[i].lower(): float(p[i]) for i in range(probs.shape[1]) }
        for p in probs
    ]

def _violence_probability_batch(pils: List[Image.Image]) -> List[float]:
    _load_violence()
    inputs = _proc_image_batch(_VIO_PROC, pils)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=_DEVICE.type=="cuda"):
        logits = _VIO_MODEL(**inputs).logits
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    violent_idx = None
    for i, lbl in _VIO_ID2LABEL.items():
        if str(lbl).lower().startswith("violent"):
            violent_idx = i
            break

    if violent_idx is not None:
        return probs[:, violent_idx].tolist()

    # Fallback: choose max per row
    row_argmax = probs.argmax(axis=1)
    return probs[np.arange(probs.shape[0]), row_argmax].tolist()

def _clip_text_features(key: str, texts: List[str]) -> Tuple[torch.Tensor, float]:
    _load_clip()
    if key in _CLIP_TXT_CACHE:
        return _CLIP_TXT_CACHE[key]
    tokens = _CLIP_PROC.tokenizer(texts, padding=True, return_tensors="pt")
    device = next(_CLIP_MODEL.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=_DEVICE.type=="cuda"):
        t = _CLIP_MODEL.get_text_features(**tokens)  # [T, D]
    # L2-normalize and **store in float32** (robust across contexts)
    t = t / t.norm(p=2, dim=-1, keepdim=True)
    t = t.float()                                     # <-- force fp32 in cache
    scale = float(_CLIP_MODEL.logit_scale.exp().detach().cpu().item())
    _CLIP_TXT_CACHE[key] = (t, scale)
    return _CLIP_TXT_CACHE[key]

def _clip_image_features(pils: List[Image.Image]):
    _load_clip()
    pixel = _CLIP_PROC(images=pils, return_tensors="pt")
    device = next(_CLIP_MODEL.parameters()).device
    pixel = {k: v.to(device) for k, v in pixel.items()}
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=_DEVICE.type=="cuda"):
        img_feats = _CLIP_MODEL.get_image_features(**pixel)  # [N, D]
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    return img_feats.float()                                 # <-- always fp32

def _clip_posmass_logodds_batch(
    pils: List[Image.Image],
    pos_txt: List[str],
    neg_txt: List[str],
    alpha: float = 1.2,  # tuning knob: >1 sharpens, <1 softens
) -> List[float]:
    """
    Binary log-odds pooling with CLIP.
    Returns P(positive-group | image) in [0,1] per image.
    Uses the same cached text features as horror.
    """
    # 1) prepare/cached text feats
    texts = pos_txt + neg_txt
    Tpos = len(pos_txt)
    key = "BIN|" + "|".join(texts)
    tfeat, scale = _clip_text_features(key, texts)  # tfeat: [T,D] (fp32, L2-norm), scale: float

    # 2) image feats (batched)
    vfeat = _clip_image_features(pils)              # [N,D], fp32, L2-norm

    # 3) logits per text, pooled by group via log-sum-exp
    scale = torch.tensor(scale, device=vfeat.device, dtype=vfeat.dtype)
    logits = scale * (vfeat @ tfeat.T)              # [N,T]
    lse_pos = torch.logsumexp(logits[:, :Tpos], dim=-1)   # [N]
    lse_neg = torch.logsumexp(logits[:, Tpos:], dim=-1)   # [N]

    # 4) binary log-odds → probability
    logit = alpha * (lse_pos - lse_neg)             # [N]
    probs = torch.sigmoid(logit).detach().cpu().numpy().tolist()
    return probs

def _horror_probability_batch(pils: List[Image.Image], alpha: float = 1.2) -> List[float]:
    texts = _HORROR_TEXTS + _SAFE_TEXTS
    Tpos = len(_HORROR_TEXTS)
    key = "HORR|" + "|".join(texts)
    tfeat, scale = _clip_text_features(key, texts)         # fp32
    vfeat = _clip_image_features(pils)                     # fp32
    scale = torch.tensor(scale, device=vfeat.device, dtype=vfeat.dtype)
    logits = scale * (vfeat @ tfeat.T)
    lse_pos = torch.logsumexp(logits[:, :Tpos], dim=-1)
    lse_neg = torch.logsumexp(logits[:, Tpos:], dim=-1)
    probs = torch.sigmoid(alpha * (lse_pos - lse_neg)).detach().cpu().numpy()
    return probs.tolist()

def _weapon_probability_batch(pils: List[Image.Image]) -> List[float]:
    return _clip_posmass_logodds_batch(pils, _WEAPON_TXT, _NEG_TXT, alpha=1.2)

def _blood_probability_batch(pils: List[Image.Image]) -> List[float]:
    return _clip_posmass_logodds_batch(pils, _BLOOD_TXT, _NEG_TXT, alpha=1.2)

# -------------------------
# Calibration helper
# -------------------------
def sweep_alpha_margin(
    pil_images,                # List[Image.Image]
    labels,                    # List[int]  (1=safe, 0=not safe)
    profile="Strict (0-6)",
    alphas=(1.0, 1.2, 1.4, 1.6),
    margins=(0.6, 0.7, 0.8),
    use_violence=True, use_horror=True, use_weap=True, use_blood=True,
    scan_tiles=True, use_smart_tiles=True,
    show_progress: bool = False,          # NEW
):
    """
    Runs a small grid search and prints accuracy/recall/precision and median runtime.
    Returns a list of tuples: [((alpha, margin), acc, recall_safe, precision_safe, median_ms), ...]
    """
    import statistics as stats
    import time

    assert len(pil_images) == len(labels), "pil_images and labels must match in length"

    # optional progress bar
    pbar = None
    total_steps = len(alphas) * len(margins) * max(1, len(pil_images))
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_steps, desc="sweep", unit="img")
        except Exception:
            pbar = None  # silently disable if tqdm isn't available

    results = []

    for a in alphas:
        for m in margins:
            preds, times = [], []
            for pil in pil_images:
                t0 = time.perf_counter()

                nsfw = _nsfw_probabilities_batch([pil])[0]
                vio  = _violence_probability_batch([pil])[0] if use_violence else 0.0
                hor  = _horror_probability_batch([pil], alpha=a)[0] if use_horror else 0.0
                wep  = _clip_posmass_logodds_batch([pil], _WEAPON_TXT, _NEG_TXT, alpha=a)[0] if use_weap else 0.0
                bld  = _clip_posmass_logodds_batch([pil], _BLOOD_TXT,  _NEG_TXT, alpha=a)[0] if use_blood else 0.0

                near = max(nsfw.get("porn",0.0)+nsfw.get("sexy",0.0), vio, hor, wep, bld)
                do_tiles = scan_tiles and (not use_smart_tiles or near >= m)

                pil_list = [pil] + (_tile_pils(pil) if do_tiles else [])
                nsfw_list = _nsfw_probabilities_batch(pil_list)
                vio_list  = _violence_probability_batch(pil_list) if use_violence else [0.0]*len(pil_list)
                hor_list  = _horror_probability_batch(pil_list, alpha=a) if use_horror else [0.0]*len(pil_list)
                wep_list  = _clip_posmass_logodds_batch(pil_list, _WEAPON_TXT, _NEG_TXT, alpha=a) if use_weap else [0.0]*len(pil_list)
                bld_list  = _clip_posmass_logodds_batch(pil_list, _BLOOD_TXT,  _NEG_TXT, alpha=a) if use_blood else [0.0]*len(pil_list)
                cos_list  = _clip_posmass_logodds_batch(pil_list, _COSMETICS_TXT, _NEG_TXT, alpha=1.0)
                sex_list  = _clip_posmass_logodds_batch(pil_list, _SEX_POS_TXT, _SEX_NEG_TXT, alpha=1.0)

                best = None
                p = PROFILES[profile]
                HARD = max(p["nsfw_sum"], p["nsfw_peak"], p["violence"], p["horror"], p["weap"], p["blood"]) + 0.20
                for i in range(len(pil_list)):
                    c, agg, extras, _ = _pg_decision(
                        nsfw_list[i], vio_list[i], hor_list[i], wep_list[i], bld_list[i], cos_list[i], sex_list[i],
                        p["nsfw_sum"], p["nsfw_peak"], p["violence"], p["horror"], p["weap"], p["blood"]
                    )
                    if (best is None) or (agg > best[1]):
                        best = (c, agg)
                        if max(extras["nsfw_sum"], vio_list[i], hor_list[i], wep_list[i], bld_list[i]) >= HARD:
                            break

                preds.append(best[0])
                times.append(time.perf_counter() - t0)

                if pbar is not None:
                    try:
                        pbar.set_postfix_str(f"a={a:.1f} m={m:.2f}")
                        pbar.update(1)
                    except Exception:
                        pass

            # metrics
            y_true = labels
            y_pred = preds
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==1)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==0)
            acc = (tp+tn)/len(y_true) if y_true else 0.0
            recall_safe = tp/(tp+fn) if (tp+fn)>0 else 0.0
            precision_safe = tp/(tp+fp) if (tp+fp)>0 else 0.0
            med_ms = 1000*stats.median(times) if times else 0.0

            results.append(((a, m), acc, recall_safe, precision_safe, med_ms))

    if pbar is not None:
        try: pbar.close()
        except Exception: pass

    # same printout as before
    print("alpha | smart_margin | acc | recall(SAFE) | precision(SAFE) | median_ms")
    results_sorted = sorted(results, key=lambda r: (-r[1], -r[2], r[4]))
    for (a, m), acc, rec, prec, ms in results_sorted:
        print(f"{a:>4.1f} | {m:>11.2f} | {acc:>4.3f} | {rec:>5.3f}       | {prec:>6.3f}          | {ms:>8.1f}")
    return results_sorted

# -------------------------
# Node
# -------------------------
class NSFWDetector_v1:
    NAME = "NSFW PG Detector"
    SHORTCUT = "i"

    """
    Inputs:
      - image: ComfyUI IMAGE tensor [B,H,W,C] or [B,C,H,W], float in [0,1]
    Outputs:
      - STRING: JSON string:
        {
          "Classification": 1|0,            # 1 = SAFE (PG), 0 = NOT SAFE
          "Score": float,                   # max risk among sexual/violence/horror (rounded to 3 decimals)
          "Reason": str,
          "Classes": {...nsfw head probs...},
          "Extras": {"nsfw_sum","nsfw_peak","violence","horror"},
          "Profile": "..."
        }
      - FLOAT: Score (same as above, rounded to 3 decimals)
      - IMAGE: passthrough
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # Dropdown profile — overrides sliders
                "profile": (list(PROFILES.keys()), {
                    "default": "Strict (0-6)"
                }),
                # Feature toggles
                "use_violence": ("BOOLEAN", {"default": True}),
                "use_horror": ("BOOLEAN", {"default": True}),
                "use_weap": ("BOOLEAN", {"default": True}),
                "use_blood": ("BOOLEAN", {"default": True}),
                "scan_tiles": ("BOOLEAN", {"default": False}),
                "use_smart_tiles": ("BOOLEAN", {"default": True}),
                "smart_margin": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Sliders (useful for calibration; overridden by profile if selected)
                "nsfw_sum_thresh": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nsfw_peak_thresh": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 1.0, "step": 0.01}),
                "violence_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "horror_thresh": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weap_thresh": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blood_thresh": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_horror": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.05}),
                "alpha_weap":   ("FLOAT", {"default": 1.2, "min": 0.8, "max": 2.0, "step": 0.05}),
                "alpha_blood":  ("FLOAT", {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("image", "safe", "reason", "json")
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")

    def fn(
        self,
        image: torch.Tensor,
        profile: str = "Strict (0-6)",
        use_violence: bool = True,
        use_horror: bool = True,
        use_weap: bool = True,
        use_blood: bool = True,
        scan_tiles: bool = False,
        use_smart_tiles: bool = True,
        smart_margin: float = 0.7,
        nsfw_sum_thresh: float = 0.02,
        nsfw_peak_thresh: float = 0.015,
        violence_thresh: float = 0.15,
        horror_thresh: float = 0.30,
        weap_thresh: float = 0.30,
        blood_thresh: float = 0.30,
        alpha_horror: float = 1.0,
        alpha_weap: float = 1.2,
        alpha_blood: float = 1.2,
    ):
        pil_batch = _tensor_to_pil_batch(image)

        # Apply profile presets (override sliders)
        if profile in PROFILES:
            preset = PROFILES[profile]
            nsfw_sum_thresh  = preset["nsfw_sum"]
            nsfw_peak_thresh = preset["nsfw_peak"]
            violence_thresh  = preset["violence"]
            horror_thresh    = preset["horror"]
            weap_thresh    = preset["weap"]
            blood_thresh    = preset["blood"]

        # If disabled, bypass model calls and make them non-operative
        if not use_violence:
            violence_thresh = 1e9  # never triggers
        if not use_horror:
            horror_thresh = 1e9    # never triggers
        if not use_weap:
            weap_thresh = 1e9  # never triggers
        if not use_blood:
            blood_thresh = 1e9    # never triggers

        results = []
        risk_scores = []

        for pil_img in pil_batch:
            # full image first
            nsfw_full = _nsfw_probabilities_batch([pil_img])[0]
            vio_full  = _violence_probability_batch([pil_img])[0] if use_violence else 0.0
            hor_full  = _horror_probability_batch([pil_img], alpha=alpha_horror)[0] if use_horror else 0.0
            weap_full = _clip_posmass_logodds_batch([pil_img], _WEAPON_TXT, _NEG_TXT, alpha=alpha_weap)[0] if use_weap else 0.0
            blood_full= _clip_posmass_logodds_batch([pil_img], _BLOOD_TXT,  _NEG_TXT, alpha=alpha_blood)[0] if use_blood else 0.0

            near = max(
                nsfw_full.get("porn",0.0) + nsfw_full.get("sexy",0.0), 
                vio_full, hor_full, weap_full, blood_full
            )
            do_tiles = scan_tiles and (not use_smart_tiles or near >= smart_margin)

            # 1) build scan list
            tiles = _tile_pils(pil_img)
            if len(tiles) > 64: tiles = tiles[:64]
            pil_list = [pil_img] + (tiles if do_tiles else [])

            # 2) batch heads
            nsfw_list = _nsfw_probabilities_batch(pil_list)            # List[dict]
            vio_list  = _violence_probability_batch(pil_list) if use_violence else [0.0]*len(pil_list)

            # 3) (for now) horror, weap, blood per-tile — will batch in step 2 below
            hor_list  = _horror_probability_batch(pil_list, alpha=alpha_horror) if use_horror else [0.0]*len(pil_list)
            weap_list = _clip_posmass_logodds_batch(pil_list, _WEAPON_TXT, _NEG_TXT, alpha=alpha_weap) if use_weap else [0.0]*len(pil_list)
            blood_list= _clip_posmass_logodds_batch(pil_list, _BLOOD_TXT,  _NEG_TXT, alpha=alpha_blood) if use_blood else [0.0]*len(pil_list)
            cos_list= _clip_posmass_logodds_batch(pil_list, _COSMETICS_TXT,  _NEG_TXT, alpha=1.0) if use_blood else [0.0]*len(pil_list)
            sex_list  = _clip_posmass_logodds_batch(pil_list, _SEX_POS_TXT, _SEX_NEG_TXT, alpha=1.0)

            # 4) pick worst tile
            best = None  # (i, nsfw_probs, vio_prob, hor_prob, classification, agg_score, extras, reason)
            hard_margin = 0.10 if "Strict" in profile else 0.20
            HARD = max(nsfw_sum_thresh, nsfw_peak_thresh, violence_thresh, horror_thresh, weap_thresh, blood_thresh) + hard_margin
            for i in range(len(pil_list)):
                nsfw_probs = nsfw_list[i]
                vio_prob   = vio_list[i]
                hor_prob   = hor_list[i]
                weap_prob  = weap_list[i]
                blood_prob = blood_list[i]
                cos_prob   = cos_list[i]
                sex_prob   = sex_list[i]

                classification, agg_score, extras, reason = _pg_decision(
                    nsfw_probs, vio_prob, hor_prob, weap_prob, blood_prob, cos_prob, sex_prob,
                    nsfw_sum_thresh, nsfw_peak_thresh, violence_thresh, horror_thresh, weap_thresh, blood_thresh
                )
                if (best is None) or (agg_score > best[7]):
                    best = (i, nsfw_probs, vio_prob, hor_prob, weap_prob, blood_prob, classification, agg_score, extras, reason)
                    if max(extras["nsfw_sum"], vio_prob, hor_prob, weap_prob, blood_prob) >= HARD:
                        break

            # 5) unpack worst and append result (INSIDE the pil_img loop)
            _, nsfw_probs, vio_prob, hor_prob, weap_prob, blood_prob, classification, agg_score, extras, reason = best
            nsfw_probs_out = fmt_max_dict3(nsfw_probs)

            results.append({
                "Classification": int(classification),
                "Score": fmt_max3(float(agg_score)),
                "Reason": reason if int(classification)==0 else "OK (Content appears kid-friendly)",
                "Classes": nsfw_probs_out,
                "Extras": {
                    "nsfw_sum": fmt_max3(extras["nsfw_sum"]),
                    "nsfw_peak": fmt_max3(extras["nsfw_peak"]),
                    "violence": fmt_max3(extras["violence"]),
                    "horror": fmt_max3(extras["horror"]),
                    "weapon": fmt_max3(extras["weapon"]),
                    "blood": fmt_max3(extras["blood"]),
                    "cosmetics": fmt_max3(extras["cosmetics"]),
                    "sexual_ctx": fmt_max3(extras["sexual_ctx"]),
                },
                "Profile": profile,
                "ScanMode": "tiles" if do_tiles else "full",
                "Tiles": len(pil_list)-1 if do_tiles else 0,
            })
            risk_scores.append(float(agg_score))


        payload = results[0] if len(results) == 1 else results
        json_text = json.dumps(payload["Extras"], ensure_ascii=False)

        # FLOAT output = max risk over batch, rounded to 3 decimals
        # score_out = fmt_max3(float(max(risk_scores)) if risk_scores else 0.000)
        safe_bool = bool(payload["Classification"] == 1)
        return (image, safe_bool, payload["Reason"], json_text)
