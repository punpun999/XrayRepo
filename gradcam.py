r"""
gradcam.py -- Grad-CAM explainability, rewritten from visualize_fracture.py.

What the old script got wrong
-----------------------------
1. It pointed at a DESKTOP SCREENSHOT as its test image. Grad-CAM on a
   non-X-ray is meaningless -- the heatmap shows where the network found
   evidence for a class it was forced to pick from six bone classes.
2. It hardcoded `models.densenet121()` and a `D:\` weights path, so it broke
   the moment you trained anything else. Now it goes through
   common.load_checkpoint(), which sniffs the architecture.
3. It hardcoded 224px. A model trained at 320 (train_v4/v5) was being fed the
   wrong resolution, which silently degrades both the prediction and the map.
4. It backpropagated from `argmax` over the six classes. On a healthy wrist
   that explains "why wrist", not "why healthy" -- so the interesting question
   (what does the model think a fracture looks like?) was never asked.
   `--target fracture` backprops from P(fracture) instead, which is the map
   you actually want.
5. `plt.show()` blocks; nothing was ever saved. Now writes PNGs.

The old `custom_forward` monkey-patch is gone, but NOT because it was pointless
-- it was working around a real autograd constraint. torchvision's DenseNet
applies `F.relu(features, inplace=True)` to norm5's output, and a full backward
hook on norm5 wraps that output in a custom autograd Function whose result is a
view; mutating it in place is forbidden. Rather than reimplement the forward
pass, this version hooks `features.denseblock4`, which feeds norm5 through an
out-of-place BatchNorm and so is never mutated. Same spatial resolution, no
divergence from torchvision's real forward.

Usage
-----
    python gradcam.py xray.png --weights v4_densenet.pth
    python gradcam.py study_folder/ --weights v5_multihead.pth --target fracture
    python gradcam.py xray.png --weights best.pth --outdir cams --no-show
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from common import (
    BROKEN_INDICES,
    MultiHeadNet,
    broken_probability,
    clean_path,
    eval_transform,
    gradcam_target_layer,
    load_checkpoint,
    pick_device,
)

DEFAULT_WEIGHTS = r"D:\archive\Project_Dataset\master_densenet_weights.pth"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------
class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Weights each channel of the target layer's feature map by the mean
    gradient of the chosen scalar with respect to that channel, sums, and
    ReLUs. The result highlights the pixels that pushed the score up.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        # Defensive copy, not the load-bearing fix. Forward hooks run BEFORE
        # register_full_backward_hook wraps the output, so a clone here cannot
        # protect against an in-place op applied downstream -- that is handled
        # by choosing a target layer nothing mutates (see
        # common.gradcam_target_layer). The clone costs one feature map and
        # makes this class safe against any target layer the caller picks.
        self.activations = output.clone()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def generate(self, input_tensor: torch.Tensor, target: str = "fracture"):
        """Return (heatmap [H, W] in [0, 1], class_index, probs [C]).

        target="fracture" backprops from P(fracture) -- for a multi-head model
        that is the fracture head's logit directly; for a 6-way model it is the
        summed log-probability of the three Broken classes. This answers "what
        made this look broken", which is the clinically interesting question.

        target="class" backprops from the top predicted class, which on a
        healthy image explains body part rather than pathology.
        """
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.requires_grad_(False)

        output = self.model(input_tensor)              # [1, C] logits or log-probs
        probs = torch.softmax(output.float(), dim=1)[0]
        pred_class = int(torch.argmax(probs).item())

        if target == "fracture":
            if isinstance(self.model, MultiHeadNet):
                # Single calibrated logit -- the cleanest possible signal.
                score = self.model.fracture_logit(input_tensor).squeeze()
            else:
                # log of the summed Broken probability. Using the log keeps the
                # gradient well-scaled when P(fracture) is near 0 or 1.
                p_broken = probs[BROKEN_INDICES].sum()
                score = torch.log(p_broken.clamp_min(1e-12))
        elif target == "class":
            score = output[0, pred_class]
        else:
            raise ValueError("target must be 'fracture' or 'class'")

        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM hooks captured nothing. The target layer is probably "
                "not on the path from input to the score you selected.")

        # Channel weights = global-average-pooled gradients.
        weights = self.gradients.mean(dim=(0, 2, 3))            # [C_feat]
        acts = self.activations.detach()[0]                     # [C_feat, H, W]
        cam = (acts * weights[:, None, None]).sum(dim=0)        # [H, W]
        cam = torch.relu(cam).cpu().numpy()

        peak = float(cam.max())
        if peak > 0:
            cam = cam / peak
        return cam, pred_class, probs.detach().cpu().numpy()


# --------------------------------------------------------------------------
def colourise(cam: np.ndarray, size) -> np.ndarray:
    """Resize the CAM to `size` = (W, H) and apply a jet-like colour map.

    Uses OpenCV when available and falls back to matplotlib, so the script does
    not hard-depend on opencv-python the way visualize_fracture.py did.
    """
    w, h = size
    try:
        import cv2
        resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        bgr = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        import matplotlib.cm as cm
        resized = np.array(
            Image.fromarray(np.float32(cam), mode="F").resize((w, h), Image.BILINEAR))
        return (cm.jet(np.clip(resized, 0, 1))[..., :3] * 255).astype(np.uint8)


def overlay(original: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat = colourise(cam, original.size).astype(np.float32)
    base = np.array(original.convert("RGB")).astype(np.float32)
    return np.clip(heat * alpha + base * (1 - alpha), 0, 255).astype(np.uint8)


def collect_images(inputs) -> list:
    paths = []
    for raw in inputs:
        item = clean_path(raw)
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for f in sorted(files):
                    if f.lower().endswith(IMAGE_EXTS) and not f.startswith("._"):
                        paths.append(os.path.join(root, f))
        else:
            paths.append(item)
    return paths


def render(original, cam, title, subtitle, out_path, show):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        Image.fromarray(overlay(original, cam)).save(out_path)
        print(f"  [i] matplotlib missing -- wrote bare overlay to {out_path}")
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original, cmap="gray")
    ax[0].set_title("Original X-ray")
    ax[0].axis("off")
    ax[1].imshow(overlay(original, cam))
    ax[1].set_title(f"{title}\n{subtitle}", fontsize=10)
    ax[1].axis("off")
    fig.suptitle("Research tool only -- not a medical device.",
                 fontsize=8, y=0.03, color="#888888")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Grad-CAM heatmaps showing which pixels drove the prediction.")
    ap.add_argument("images", nargs="+",
                    help="Image file(s) or folder(s). Point this at a real MURA "
                         "X-ray -- a heatmap over anything else is noise.")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--arch", default=None, help="Normally inferred from the checkpoint.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--size", type=int, default=None,
                    help="Input resolution. Defaults to the checkpoint's own "
                         "(224 for legacy checkpoints).")
    ap.add_argument("--target", default="fracture", choices=("fracture", "class"),
                    help="What to explain. 'fracture' (default) asks what made "
                         "this look broken; 'class' explains the top-1 label, "
                         "which on a healthy image just explains body part.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(fracture) above which to label the image as flagged. "
                         "Use the value evaluate.py recommends, not 0.5.")
    ap.add_argument("--outdir", default="gradcam_out")
    ap.add_argument("--alpha", type=float, default=0.4, help="Heatmap opacity.")
    ap.add_argument("--no-show", action="store_true",
                    help="Write PNGs without opening a window (default when "
                         "more than one image is given).")
    args = ap.parse_args(argv)

    paths = collect_images(args.images)
    if not paths:
        print("No images matched the given input.", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    device = pick_device(args.device)
    print(f"Device: {device}")

    try:
        ckpt = load_checkpoint(clean_path(args.weights), device=device, arch=args.arch)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[!] Could not load model: {exc}", file=sys.stderr)
        return 1

    size = args.size or ckpt.size or 224
    print(f"Model:  {ckpt.arch} @ {size}px   explaining: {args.target}")

    layer = gradcam_target_layer(ckpt.model)
    tf = eval_transform(size)
    show = (not args.no_show) and len(paths) == 1

    with GradCAM(ckpt.model, layer) as cam_engine:
        for path in paths:
            try:
                original = Image.open(path).convert("RGB")
            except Exception as exc:
                print(f"[!] Skipping {path}: {exc}", file=sys.stderr)
                continue

            tensor = tf(original).unsqueeze(0).to(device)
            cam, pred, probs = cam_engine.generate(tensor, target=args.target)

            p_fx = broken_probability(probs)
            flag = "FRACTURE SUSPECTED" if p_fx >= args.threshold else "no fracture flagged"
            title = f"Predicted: {ckpt.classes[pred]}"
            subtitle = (f"P(fracture) = {p_fx * 100:.1f}%  "
                        f"@ threshold {args.threshold:.2f} -> {flag}")

            # MURA names every image image1/2/3.png inside its own study
            # folder, so basename alone collides across patients and silently
            # overwrites earlier heatmaps. Qualify it with patient + study.
            parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
            stem = os.path.splitext("_".join(parts[-3:]))[0]
            out_path = os.path.join(args.outdir, f"{stem}_gradcam_{args.target}.png")
            render(original, cam, title, subtitle, out_path, show)

            print(f"  {os.path.basename(path):<40s} {ckpt.classes[pred]:<18s} "
                  f"P(fx)={p_fx * 100:5.1f}%  -> {out_path}")

    print(f"\nWrote {len(paths)} heatmap(s) to {args.outdir}")
    print("Research tool only. Not a medical device; not for clinical use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
