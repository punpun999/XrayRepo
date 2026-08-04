"""
predict.py -- run the trained model on one or more X-ray images.

FIXED: the previous version hardcoded `models.resnet18(...)` and loaded
custom_xray_resnet.pth. FineTune_DL_v2.py had already overwritten that file
with DenseNet121 weights, so load_state_dict raised a key mismatch. The
architecture is now inferred from the checkpoint itself (see
common.load_checkpoint), and paths come from the command line.

Usage
-----
    python predict.py path/to/xray.png
    python predict.py img1.png img2.png --weights D:\\archive\\Project_Dataset\\master_densenet_weights.pth
    python predict.py folder/ --weights best.pth --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from common import (
    BROKEN_INDICES,
    broken_probability,
    clean_path,
    eval_transform,
    load_checkpoint,
    pick_device,
    predict_tensor,
)
from PIL import Image

DEFAULT_WEIGHTS = r"D:\archive\Project_Dataset\master_densenet_weights.pth"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_images(inputs) -> list:
    """Expand files, folders and globs into a flat list of image paths."""
    paths = []
    for raw in inputs:
        item = clean_path(raw)
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for f in sorted(files):
                    if f.lower().endswith(IMAGE_EXTS) and not f.startswith("._"):
                        paths.append(os.path.join(root, f))
        elif any(ch in item for ch in "*?["):
            paths.extend(sorted(glob.glob(item)))
        else:
            paths.append(item)
    return paths


def report(path, probs, classes, quiet=False, threshold=0.5) -> dict:
    """Print a diagnostic report and return it as a dict."""
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    p_broken = broken_probability(probs)
    flag = p_broken >= threshold

    result = {
        "file": path,
        "prediction": classes[top],
        "confidence": float(probs[top]) * 100.0,
        "p_fracture": p_broken * 100.0,
        "threshold": threshold,
        "fracture_flag": bool(flag),
        "probabilities": {c: float(p) * 100.0 for c, p in zip(classes, probs)},
    }

    if quiet:
        return result

    print("\n" + "=" * 52)
    print("               DIAGNOSTIC REPORT")
    print("=" * 52)
    print(f"File:        {os.path.basename(path)}")
    print("-" * 52)
    print(f"Prediction:  {result['prediction'].upper()}")
    print(f"Confidence:  {result['confidence']:.2f}%")
    print(f"P(fracture): {result['p_fracture']:.2f}%   "
          f"<- summed over {len(BROKEN_INDICES)} 'Broken' classes")
    print(f"At threshold {threshold:.3f}:  "
          f"{'FRACTURE SUSPECTED' if flag else 'no fracture flagged'}")
    print("-" * 52)
    print("Full probability distribution:")
    for i in order:
        marker = "*" if i == top else " "
        print(f" {marker} {classes[i]:<20s} {probs[i] * 100:6.2f}%")
    print("=" * 52)
    print("\nResearch tool only. Not a medical device; not for clinical use.")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Classify musculoskeletal X-rays.")
    ap.add_argument("images", nargs="+",
                    help="Image file(s), folder(s) or glob pattern(s).")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="Path to the .pth checkpoint.")
    ap.add_argument("--arch", default=None,
                    help="Force the backbone (resnet18/resnet50/densenet121). "
                         "Normally inferred from the checkpoint.")
    ap.add_argument("--device", default=None, help="cuda / xpu / cpu.")
    ap.add_argument("--size", type=int, default=None,
                    help="Input resolution. Defaults to the checkpoint's "
                         "training resolution (224 for legacy checkpoints).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(fracture) above which to flag. Get the right value "
                         "for your model from evaluate.py's recommended "
                         "operating point -- 0.5 usually under-calls fractures.")
    ap.add_argument("--tta", action="store_true",
                    help="Average predictions over the horizontal mirror.")
    ap.add_argument("--json", default=None, help="Write all results to a JSON file.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-image reports (use with --json).")
    args = ap.parse_args(argv)

    paths = collect_images(args.images)
    if not paths:
        print("No images matched the given input.", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    print(f"Device: {device}")

    try:
        ckpt = load_checkpoint(clean_path(args.weights), device=device, arch=args.arch)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\n[!] Could not load model: {exc}", file=sys.stderr)
        return 1

    size = args.size or ckpt.size or 224
    print(f"Loaded {ckpt.arch} ({ckpt.num_classes} classes) @ {size}px "
          f"from {os.path.basename(args.weights)}")

    tf = eval_transform(size)
    results = []

    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[!] Skipping {path}: {exc}", file=sys.stderr)
            continue
        probs = predict_tensor(ckpt.model, tf(image), device, tta=args.tta)
        results.append(report(path, probs, ckpt.classes, quiet=args.quiet,
                              threshold=args.threshold))

    if args.quiet:
        for r in results:
            flag = "FRACTURE" if r["fracture_flag"] else "  clear "
            print(f"{flag}  {r['prediction']:<20s} {r['confidence']:6.2f}%  "
                  f"P(fx)={r['p_fracture']:6.2f}%  {os.path.basename(r['file'])}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {len(results)} result(s) to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
