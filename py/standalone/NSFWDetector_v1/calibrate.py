# --- import the node as a package so relative imports work ---
import sys, importlib, json, csv, re
from pathlib import Path
import argparse
import logging
import random
import datetime as dt
from typing import List, Tuple
import numpy as np

from PIL import Image, UnidentifiedImageError

HERE = Path(__file__).resolve().parent          # .../py/standalone/NSFWDetector_v1
PY_ROOT = HERE.parents[2]                       # .../py
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))            # make 'nodes' importable

# Import your node as a real package module (since /py is on sys.path)
mod = importlib.import_module("py.nodes.Info.NSFWDetector_v1")
PROFILES = mod.PROFILES
sweep_alpha_margin = mod.sweep_alpha_margin
NODE_FILE = Path(mod.__file__).resolve()

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("calibrate")

# ------------------------------------------------------------------
# Dataset utils
# ------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def compute_data_driven_thresholds_from_sfw(pil_imgs, q_sum=0.95, q_peak=0.95,
                                            q_vio=0.99, q_hor=0.99, q_wep=0.99, q_bld=0.99):
    import numpy as np
    sums, peaks, vio, hor, wep, bld = [], [], [], [], [], []
    for img in pil_imgs:
        nsfw = mod._nsfw_probabilities_batch([img])[0]
        sums.append(nsfw.get("porn",0)+nsfw.get("sexy",0)+nsfw.get("hentai",0))
        peaks.append(max(nsfw.get("porn",0), nsfw.get("sexy",0), nsfw.get("hentai",0)))
        vio.append(mod._violence_probability_batch([img])[0])
        hor.append(mod._horror_probability_batch([img])[0])
        wep.append(mod._clip_posmass_logodds_batch([img], mod._WEAPON_TXT, mod._NEG_TXT)[0])
        bld.append(mod._clip_posmass_logodds_batch([img], mod._BLOOD_TXT,  mod._NEG_TXT)[0])

    thresholds = {
        "nsfw_sum":  float(np.quantile(sums,  q_sum)),
        "nsfw_peak": float(np.quantile(peaks, q_peak)),
        "violence":  float(np.quantile(vio,   q_vio)),
        "horror":    float(np.quantile(hor,   q_hor)),
        "weap":      float(np.quantile(wep,   q_wep)),
        "blood":     float(np.quantile(bld,   q_bld)),
    }
    return thresholds

def dump_head_quantiles(pil_imgs, labels, title="ALL"):
    # compute NSFW sums/peaks + CLIP heads for each image once
    sums, peaks, vio, hor, wep, bld = [], [], [], [], [], []
    for img in pil_imgs:
        nsfw = mod._nsfw_probabilities_batch([img])[0]
        ssum  = nsfw.get("porn",0)+nsfw.get("sexy",0)+nsfw.get("hentai",0)
        speak = max(nsfw.get("porn",0), nsfw.get("sexy",0), nsfw.get("hentai",0))
        v = mod._violence_probability_batch([img])[0]
        h = mod._horror_probability_batch([img])[0]
        w = mod._clip_posmass_logodds_batch([img], mod._WEAPON_TXT, mod._NEG_TXT)[0]
        b = mod._clip_posmass_logodds_batch([img], mod._BLOOD_TXT,  mod._NEG_TXT)[0]
        sums.append(ssum)
        peaks.append(speak)
        vio.append(v)
        hor.append(h)
        wep.append(w)
        bld.append(b)

    def q(a):
        return {k: float(np.quantile(a, k)) for k in (0.5, 0.9, 0.95, 0.99)}
    log.info("\n[%s] quantiles", title)
    log.info(" nsfw_sum : %s", q(sums))
    log.info(" nsfw_peak: %s", q(peaks))
    log.info(" violence : %s", q(vio))
    log.info(" horror   : %s", q(hor))
    log.info(" weapon   : %s", q(wep))
    log.info(" blood    : %s", q(bld))

def discover_images(root: Path) -> List[Path]:
    if not root.exists():
        log.warning("Directory not found: %s", root)
        return []
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]

def load_pil_safe(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.verify()                      # basic header check
        img = Image.open(path).convert("RGB")  # reopen usable image
        return img
    except (UnidentifiedImageError, OSError) as e:
        log.warning("Skipping unreadable image: %s (%s)", path, e)
        return None

def build_dataset(
    sfw_dir: Path,
    nsfw_dir: Path,
    max_per_class: int | None = None,
    seed: int = 42,
) -> Tuple[List[Image.Image], List[int]]:
    random.seed(seed)
    sfw_paths = discover_images(sfw_dir)
    nsfw_paths = discover_images(nsfw_dir)

    if max_per_class is not None:
        random.shuffle(sfw_paths)
        random.shuffle(nsfw_paths)
        sfw_paths = sfw_paths[:max_per_class]
        nsfw_paths = nsfw_paths[:max_per_class]

    pil_imgs: List[Image.Image] = []
    labels: List[int] = []

    for p in sfw_paths:
        img = load_pil_safe(p)
        if img is not None:
            pil_imgs.append(img)
            labels.append(1)

    for p in nsfw_paths:
        img = load_pil_safe(p)
        if img is not None:
            pil_imgs.append(img)
            labels.append(0)

    idx = list(range(len(pil_imgs)))
    random.shuffle(idx)
    pil_imgs = [pil_imgs[i] for i in idx]
    labels   = [labels[i]   for i in idx]

    log.info("Loaded %d SFW and %d NSFW (after filtering). Total=%d",
             sum(labels), len(labels) - sum(labels), len(labels))
    return pil_imgs, labels

# ------------------------------------------------------------------
# Path resolution (relative to calibrate.py when not absolute)
# ------------------------------------------------------------------
def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (HERE / p).resolve()

# ------------------------------------------------------------------
# Profile normalization
# ------------------------------------------------------------------
PROFILE_ALIASES = {
    "strict (0-6)": "Strict (0–6)",
    "strict (06)": "Strict (0–6)",
    "strict-0-6": "Strict (0–6)",
    "moderate (7-9)": "Moderate (7–9)",
    "older kids (10-12)": "Older kids (10–12)",
}

def normalize_profile(s: str) -> str:
    key = s.strip().lower().replace("–", "-").replace("—", "-")
    return PROFILE_ALIASES.get(key, s)

def sanitize_tag(s: str) -> str:
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "", s)

# ------------------------------------------------------------------
# Save results into ALPHA/
# ------------------------------------------------------------------
def save_results_to_alpha(alpha_dir: Path, profile: str, alphas, margins, args_dict, results):
    alpha_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = sanitize_tag(profile)
    base = f"sweep_{tag}_{ts}"

    # CSV
    csv_path = alpha_dir / f"{base}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "smart_margin", "accuracy", "recall_safe", "precision_safe", "median_ms"])
        for (a, m), acc, rec, prec, ms in results:
            w.writerow([a, m, f"{acc:.6f}", f"{rec:.6f}", f"{prec:.6f}", f"{ms:.1f}"])

    # JSON metadata
    json_path = alpha_dir / f"{base}.json"
    meta = dict(
        profile=profile,
        alphas=list(alphas),
        margins=list(margins),
        args=args_dict,
        node_file=str(NODE_FILE),
        timestamp=ts,
    )
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Best pick (same ordering as sweep sort)
    best = sorted(results, key=lambda r: (-r[1], -r[2], r[4]))[0]
    best_path = alpha_dir / f"{base}_best.txt"
    (alpha, margin), acc, rec, prec, ms = best
    best_path.write_text(
        f"Best @ profile={profile}\nalpha={alpha}, smart_margin={margin}\n"
        f"accuracy={acc:.6f}, recall_safe={rec:.6f}, precision_safe={prec:.6f}, median_ms={ms:.1f}\n",
        encoding="utf-8"
    )
    log.info("Saved:\n  %s\n  %s\n  %s", csv_path, json_path, best_path)

def save_calibrated_profile(alpha_dir: Path, profile_name: str, thresholds: dict):
    alpha_dir.mkdir(parents=True, exist_ok=True)
    out = {"Calibrated (SFW-quantiles)": thresholds}
    path = alpha_dir / "profiles_calibrated.json"
    import json
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Saved calibrated profile -> %s", path)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sweep alpha & smart_margin for NSFW PG detector.")
    parser.add_argument("--sfw",  default="SFW",  help="Path to SFW folder (default: ./SFW)")
    parser.add_argument("--nsfw", default="NSFW", help="Path to NSFW folder (default: ./NSFW)")
    parser.add_argument("--profile", default="Strict (0–6)",
                        help="Age profile ('Strict (0-6)', 'Moderate (7-9)', 'Older kids (10-12)')")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap number of images per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=str, default="1.0,1.2,1.4,1.6",
                        help="Comma-separated alpha values")
    parser.add_argument("--margins", type=str, default="0.6,0.7,0.8",
                        help="Comma-separated smart_margin values")
    # Feature toggles
    parser.add_argument("--use-violence", action="store_true", default=True)
    parser.add_argument("--no-violence",  dest="use_violence", action="store_false")
    parser.add_argument("--use-horror",   action="store_true", default=True)
    parser.add_argument("--no-horror",    dest="use_horror",   action="store_false")
    parser.add_argument("--use-weap",     action="store_true", default=True)
    parser.add_argument("--no-weap",      dest="use_weap",     action="store_false")
    parser.add_argument("--use-blood",    action="store_true", default=True)
    parser.add_argument("--no-blood",     dest="use_blood",    action="store_false")
    parser.add_argument("--scan-tiles",   action="store_true", default=True)
    parser.add_argument("--no-scan-tiles", dest="scan_tiles",  action="store_false")
    parser.add_argument("--use-smart-tiles", action="store_true", default=True)
    parser.add_argument("--no-smart-tiles",  dest="use_smart_tiles", action="store_false")

    args = parser.parse_args(argv)

    args.profile = normalize_profile(args.profile)
    if args.profile not in PROFILES:
        raise SystemExit(f"Unknown profile '{args.profile}'. Choices: {list(PROFILES.keys())}")

    # Resolve paths relative to this file if not absolute
    sfw_dir  = resolve_path(args.sfw)
    nsfw_dir = resolve_path(args.nsfw)

    alphas  = tuple(float(x.strip()) for x in args.alphas.split(",") if x.strip())
    margins = tuple(float(x.strip()) for x in args.margins.split(",") if x.strip())

    pil_imgs, labels = build_dataset(sfw_dir, nsfw_dir, args.max_per_class, args.seed)
    if not pil_imgs:
        log.error("No images loaded. Check your SFW/NSFW folders.")
        return 1

    log.info("Node module file: %s", NODE_FILE)
    log.info("Running sweep — profile=%s, alphas=%s, margins=%s", args.profile, alphas, margins)

    # ---- run sweep WITH progress bar ----
    results = sweep_alpha_margin(
        pil_imgs,
        labels,
        profile=args.profile,
        alphas=alphas,
        margins=margins,
        use_violence=args.use_violence,
        use_horror=args.use_horror,
        use_weap=args.use_weap,
        use_blood=args.use_blood,
        scan_tiles=args.scan_tiles,
        use_smart_tiles=args.use_smart_tiles,
        show_progress=True,       # NEW
    )

    # ---- export to ALPHA/ next to this script ----
    alpha_dir = HERE / "ALPHA"
    args_dict = {
        "sfw": str(sfw_dir),
        "nsfw": str(nsfw_dir),
        "max_per_class": args.max_per_class,
        "seed": args.seed,
        "use_violence": args.use_violence,
        "use_horror": args.use_horror,
        "use_weap": args.use_weap,
        "use_blood": args.use_blood,
        "scan_tiles": args.scan_tiles,
        "use_smart_tiles": args.use_smart_tiles,
    }
    # Split by label to see SFW vs NSFW distributions
    sfw_imgs = [img for img,l in zip(pil_imgs, labels) if l==1]
    nsfw_imgs= [img for img,l in zip(pil_imgs, labels) if l==0]
    dump_head_quantiles(sfw_imgs, labels=None, title="SFW only")
    dump_head_quantiles(nsfw_imgs, labels=None, title="NSFW only")

    calib = compute_data_driven_thresholds_from_sfw(sfw_imgs,
        q_sum=0.95, q_peak=0.95, q_vio=0.99, q_hor=0.99, q_wep=0.99, q_bld=0.99)
    log.info("\n[Calibrated thresholds from SFW quantiles]")
    for k,v in calib.items(): log.info(f" {k}: {v:.3f}")
    save_calibrated_profile(alpha_dir, "Calibrated (SFW-quantiles)", calib)
    save_results_to_alpha(alpha_dir, args.profile, alphas, margins, args_dict, results)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
