"""
predict_interactive.py -- REPL for classifying X-rays one at a time.

FIXED: same architecture-mismatch bug as predict.py. The model shell is now
built from whatever the checkpoint actually contains, so it works with both
the old ResNet18 weights and the DenseNet121 ones.

Usage
-----
    python predict_interactive.py
    python predict_interactive.py --weights path/to/best.pth

Commands inside the prompt:
    <path>    classify an image (Windows "Copy as path" quotes are handled)
    model     show which checkpoint is loaded
    help      show commands
    quit      exit
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

from common import (
    broken_probability,
    clean_path,
    eval_transform,
    load_checkpoint,
    pick_device,
    predict_tensor,
)

DEFAULT_WEIGHTS = r"D:\archive\Project_Dataset\master_densenet_weights.pth"

BANNER = r"""
 ============================================================
   MURA X-RAY CLASSIFIER  --  interactive mode
   Research tool only. Not a medical device.
 ============================================================
"""

HELP = """
Commands:
  <path to image>   classify that X-ray
  model             show the loaded checkpoint
  help              show this message
  quit / exit / q   leave
"""


def show(path, probs, classes, threshold=0.5) -> None:
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    p_broken = broken_probability(probs)

    print("\n" + "=" * 52)
    print("               DIAGNOSTIC REPORT")
    print("=" * 52)
    print(f"File:        {os.path.basename(path)}")
    print("-" * 52)
    print(f"Prediction:  {classes[top].upper()}")
    print(f"Confidence:  {probs[top] * 100:.2f}%")
    print(f"P(fracture): {p_broken * 100:.2f}%")
    print(f"At threshold {threshold:.3f}:  "
          f"{'FRACTURE SUSPECTED' if p_broken >= threshold else 'no fracture flagged'}")
    print("-" * 52)
    print("Full probability distribution:")
    for i in order:
        marker = "*" if i == top else " "
        bar = "#" * int(round(probs[i] * 30))
        print(f" {marker} {classes[i]:<20s} {probs[i] * 100:6.2f}%  {bar}")
    print("=" * 52 + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Interactive X-ray classifier.")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--arch", default=None,
                    help="Force backbone; normally inferred from the checkpoint.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--size", type=int, default=None,
                    help="Defaults to the checkpoint's training resolution.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(fracture) above which to flag. Use the recommended "
                         "operating point from evaluate.py.")
    ap.add_argument("--tta", action="store_true",
                    help="Average over the horizontal mirror.")
    args = ap.parse_args(argv)

    device = pick_device(args.device)

    try:
        ckpt = load_checkpoint(clean_path(args.weights), device=device, arch=args.arch)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[!] Could not load model: {exc}", file=sys.stderr)
        print("\nHint: pass --weights to point at an existing .pth file.",
              file=sys.stderr)
        return 1

    size = args.size or ckpt.size or 224
    print(BANNER)
    print(f"Model:   {ckpt.arch}  ({ckpt.num_classes} classes) @ {size}px")
    print(f"Weights: {args.weights}")
    print(f"Device:  {device}   threshold: {args.threshold:.3f}"
          + ("   TTA: on" if args.tta else ""))
    print(HELP)

    tf = eval_transform(size)

    while True:
        try:
            raw = input("x-ray> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in {"quit", "exit", "q"}:
            print("Bye.")
            return 0
        if cmd == "help":
            print(HELP)
            continue
        if cmd == "model":
            print(f"  {ckpt.arch} | {ckpt.num_classes} classes | {args.weights}")
            print(f"  classes: {', '.join(ckpt.classes)}")
            continue

        path = clean_path(raw)
        if not os.path.isfile(path):
            print(f"[!] Not a file: {path}\n")
            continue

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[!] Could not read image: {exc}\n")
            continue

        probs = predict_tensor(ckpt.model, tf(image), device, tta=args.tta)
        show(path, probs, ckpt.classes, threshold=args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
