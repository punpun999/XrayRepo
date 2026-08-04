"""
common.py -- shared plumbing for the MURA fracture-classification project.

Everything that used to be copy-pasted (and drift out of sync) between
predict.py, predict_interactive.py, visualize_fracture.py and the training
scripts now lives here.

Key fix vs the original code:
    The old predict scripts hardcoded `resnet18`, but FineTune_DL_v2.py
    overwrote custom_xray_resnet.pth with DenseNet121 weights. Loading then
    blew up with a state_dict key mismatch. `load_checkpoint()` below sniffs
    the architecture out of the state_dict, so a checkpoint always loads into
    the right shell regardless of which training script produced it.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

# --------------------------------------------------------------------------
# Class definitions
# --------------------------------------------------------------------------
# This ordering is load-bearing. It is simultaneously:
#   1. the alphabetical order ImageFolder assigns to the folders created by
#      MainFile.py (Hand_Broken, Hand_Healthy, Shoulder_*, Wrist_*), and
#   2. the explicit class_map in Train_Full_Dataset.py's MURADataset.
# Do not reorder without retraining.
CLASSES: List[str] = [
    "Hand_Broken",
    "Hand_Healthy",
    "Shoulder_Broken",
    "Shoulder_Healthy",
    "Wrist_Broken",
    "Wrist_Healthy",
]

NUM_CLASSES = len(CLASSES)

# Derived views on the label space, used by evaluate.py to decompose the
# 6-way accuracy into its easy part (body part) and its hard part (fracture).
BODY_PARTS: List[str] = ["Hand", "Shoulder", "Wrist"]
PART_OF: List[int] = [BODY_PARTS.index(c.split("_")[0]) for c in CLASSES]
IS_BROKEN: List[int] = [1 if c.endswith("_Broken") else 0 for c in CLASSES]
BROKEN_INDICES: List[int] = [i for i, b in enumerate(IS_BROKEN) if b == 1]

# MURA raw-directory names -> our class index.
MURA_CLASS_MAP: Dict[str, int] = {
    "XR_HAND_positive": 0,
    "XR_HAND_negative": 1,
    "XR_SHOULDER_positive": 2,
    "XR_SHOULDER_negative": 3,
    "XR_WRIST_positive": 4,
    "XR_WRIST_negative": 5,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every RNG the project touches.

    The original scripts seeded nothing, so `random.sample` in MainFile.py and
    `random_split` in the fine-tune scripts gave a different dataset on every
    run -- which makes comparing v1 against v2 meaningless.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------
def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available device.

    Guards `torch.xpu` behind hasattr -- Train_Full_Dataset.py called
    torch.xpu.is_available() unconditionally, which raises AttributeError on
    any PyTorch build without Intel XPU support (i.e. most of them).
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
SUPPORTED_ARCHS = ("resnet18", "resnet50", "densenet121")
MULTIHEAD_PREFIX = "multihead:"


def build_model(arch: str, num_classes: int = NUM_CLASSES,
                pretrained: bool = False) -> nn.Module:
    """Build a classification backbone with the head resized to num_classes.

    `arch` may also be prefixed with `multihead:` (e.g. `multihead:densenet121`)
    to get a MultiHeadNet -- shared backbone, separate body-part and fracture
    heads. See that class for why.
    """
    arch = arch.lower()
    if arch.startswith(MULTIHEAD_PREFIX):
        return MultiHeadNet(arch[len(MULTIHEAD_PREFIX):], pretrained=pretrained,
                            num_classes=num_classes)
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch {arch!r}. Expected one of {SUPPORTED_ARCHS}.")
    return model


def infer_arch(state_dict: Dict[str, torch.Tensor]) -> str:
    """Guess the architecture from the shape of a bare state_dict.

    Lets us load the legacy checkpoints (`custom_xray_resnet.pth`,
    `master_densenet_weights.pth`) that were saved as raw state_dicts with no
    metadata about what produced them.
    """
    keys = set(state_dict.keys())

    # Multi-head checkpoints nest everything under `backbone.` and carry two
    # heads. Strip the prefix and recurse to identify the backbone itself.
    if "part_head.weight" in keys and "fx_head.weight" in keys:
        inner = {k[len("backbone."):]: v for k, v in state_dict.items()
                 if k.startswith("backbone.")}
        return MULTIHEAD_PREFIX + infer_arch(inner)

    if any(k.startswith("features.denseblock") for k in keys):
        return "densenet121"
    # layer4.0.conv1.weight is 3x3 in resnet18 (BasicBlock) and 1x1 in
    # resnet50 (Bottleneck) -- a cheap way to tell the two apart. Keyed off the
    # conv rather than `fc.weight` so it also works for a headless backbone.
    w = state_dict.get("layer4.0.conv1.weight")
    if w is not None:
        return "resnet50" if w.shape[-1] == 1 else "resnet18"
    raise ValueError(
        "Could not infer architecture from checkpoint. "
        "Pass --arch explicitly."
    )


def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop a `module.` prefix left behind by DataParallel, if present."""
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


@dataclass
class Checkpoint:
    model: nn.Module
    classes: List[str]
    arch: str
    device: torch.device
    size: Optional[int] = None       # input resolution the model was trained at
    meta: Optional[dict] = None      # any extra fields saved alongside

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def save_checkpoint(path: str, model: nn.Module, arch: str,
                    classes: Sequence[str] = CLASSES, **extra) -> None:
    """Save a *self-describing* checkpoint.

    New training runs should use this instead of a bare
    `torch.save(model.state_dict(), ...)`. It records the architecture and the
    class ordering alongside the weights, which is exactly the metadata whose
    absence broke the predict scripts.
    """
    payload = {
        "arch": arch,
        "classes": list(classes),
        "state_dict": model.state_dict(),
    }
    payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, device: Optional[torch.device] = None,
                    arch: Optional[str] = None) -> Checkpoint:
    """Load either a self-describing checkpoint or a legacy bare state_dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at: {path}")

    device = device or pick_device()

    # weights_only=True is the safe default in torch >= 2.6; fall back for
    # older versions that do not know the kwarg.
    try:
        blob = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location=device)

    size = None
    meta = None
    if isinstance(blob, dict) and "state_dict" in blob:
        state_dict = _strip_prefixes(blob["state_dict"])
        classes = list(blob.get("classes", CLASSES))
        arch = arch or blob.get("arch") or infer_arch(state_dict)
        size = blob.get("size")
        meta = {k: v for k, v in blob.items()
                if k not in ("state_dict", "classes", "arch")}
    else:
        # Legacy bare state_dict: no metadata, so resolution is unknown and
        # callers fall back to their default (224, what v1-v3 trained at).
        state_dict = _strip_prefixes(blob)
        classes = list(CLASSES)
        arch = arch or infer_arch(state_dict)

    model = build_model(arch, num_classes=len(classes), pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match a {arch} with {len(classes)} classes.\n"
            f"  missing keys:    {sorted(missing)[:5]}\n"
            f"  unexpected keys: {sorted(unexpected)[:5]}\n"
            "If this checkpoint came from a different backbone, pass --arch."
        )

    model = model.to(device)
    model.eval()
    return Checkpoint(model=model, classes=classes, arch=arch, device=device,
                      size=size, meta=meta)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def eval_transform(size: int = 224) -> transforms.Compose:
    """Deterministic preprocessing. Use for validation, test and inference.

    The v1/v2 fine-tune scripts wrapped a single augmented ImageFolder in
    random_split, so validation images were randomly flipped/rotated/jittered.
    Validation must be deterministic or the metric is noise.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(size: int = 224) -> transforms.Compose:
    """Augmented preprocessing. Training only.

    Horizontal flip is legitimate here: MURA contains both left and right
    limbs, so laterality carries no diagnostic signal.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class MURADataset(Dataset):
    """MURA loader for the three body parts this project uses.

    Adds `study_ids` on top of the version in Train_Full_Dataset.py. MURA's
    official benchmark scores *studies*, not images -- a study is one patient's
    set of views of one limb, and the radiologist label applies to the study as
    a whole. evaluate.py uses this to report study-level metrics that are
    comparable to published MURA numbers.

    Expected layout:
        <root>/XR_WRIST/patient00011/study1_positive/image1.png
    """

    PARTS = ("XR_HAND", "XR_SHOULDER", "XR_WRIST")

    def __init__(self, root_dir: str, transform=None, verbose: bool = True):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.study_ids: List[str] = []

        if verbose:
            print(f"Scanning {root_dir} ...")

        for subdir, _, files in os.walk(root_dir):
            part = next((p for p in self.PARTS if p in subdir), None)
            if part is None:
                continue

            lowered = subdir.replace("\\", "/").lower()
            if "positive" in lowered:
                condition = "positive"
            elif "negative" in lowered:
                condition = "negative"
            else:
                continue  # not a study folder

            label = MURA_CLASS_MAP[f"{part}_{condition}"]
            # The study folder itself is a stable, unique study identifier.
            study_id = os.path.relpath(subdir, root_dir).replace("\\", "/")

            for file in sorted(files):
                # '._' files are macOS resource forks shipped inside the
                # MURA zip; PIL chokes on them.
                if file.startswith("._"):
                    continue
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                self.image_paths.append(os.path.join(subdir, file))
                self.labels.append(label)
                self.study_ids.append(study_id)

        if not self.image_paths:
            raise RuntimeError(
                f"Found no images under {root_dir}. "
                "Check that the path points at MURA-v1.1/train or /valid."
            )

        if verbose:
            print(f"  {len(self.image_paths)} images across "
                  f"{len(set(self.study_ids))} studies")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], idx

    def class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in CLASSES}
        for lab in self.labels:
            counts[CLASSES[lab]] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...).

        MURA skews negative; none of the original training scripts corrected
        for that.
        """
        counts = np.bincount(self.labels, minlength=NUM_CLASSES).astype(np.float64)
        counts[counts == 0] = 1.0
        w = counts.sum() / (NUM_CLASSES * counts)
        return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_tensor(model: nn.Module, tensor: torch.Tensor,
                   device: torch.device, tta: bool = False) -> np.ndarray:
    """Softmax probabilities for a single preprocessed image tensor.

    With tta=True, averages over the image and its horizontal mirror. MURA
    contains both left and right limbs, so the flip preserves the label.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    probs = torch.softmax(model(tensor), dim=1)
    if tta:
        probs = (probs + torch.softmax(model(torch.flip(tensor, dims=[3])), dim=1)) / 2.0
    return probs[0].cpu().numpy()


def load_image(path: str, size: int = 224) -> Tuple[Image.Image, torch.Tensor]:
    """Open an image and return (PIL image, preprocessed tensor)."""
    image = Image.open(path).convert("RGB")
    return image, eval_transform(size)(image)


def broken_probability(probs: np.ndarray) -> float:
    """Collapse the 6-way distribution to P(fracture).

    This is the number that actually matters clinically -- the 6-way accuracy
    is dominated by body-part identification, which is trivially easy.
    """
    return float(np.asarray(probs)[BROKEN_INDICES].sum())


def clean_path(raw: str) -> str:
    """Strip the quotes Windows 'Copy as path' wraps around a path."""
    return raw.strip().strip('"').strip("'")


# --------------------------------------------------------------------------
# Study-level aggregation
# --------------------------------------------------------------------------
# MURA labels *studies*, not images: a study is one patient's set of views of
# one limb, and the radiologist's label applies to the whole study. Published
# results average P(fracture) across a study's images before scoring, so
# image-level numbers are not comparable to the literature.
#
# This used to be implemented twice -- once in evaluate.py section 5, once as
# the selection metric in train_v4.py -- with no guarantee the two agreed.
def aggregate_studies(study_ids, scores, labels=None, parts=None):
    """Mean-pool per-image fracture scores up to the study level.

    Returns (sids, mean_scores, study_labels, study_parts) with the last two
    set to None if the corresponding input was not supplied. `sids` is sorted,
    so the ordering is deterministic across runs.
    """
    from collections import defaultdict

    bucket = defaultdict(list)
    lab_of, part_of_sid = {}, {}
    for i, sid in enumerate(study_ids):
        bucket[sid].append(float(scores[i]))
        if labels is not None:
            lab_of[sid] = int(labels[i])
        if parts is not None:
            part_of_sid[sid] = int(parts[i])

    sids = sorted(bucket)
    mean_scores = np.array([float(np.mean(bucket[s])) for s in sids])
    out_labels = np.array([lab_of[s] for s in sids]) if labels is not None else None
    out_parts = np.array([part_of_sid[s] for s in sids]) if parts is not None else None
    return sids, mean_scores, out_labels, out_parts


def study_level_auc(probs, labels, study_ids,
                    broken_idx: Optional[Sequence[int]] = None,
                    is_broken: Optional[Sequence[int]] = None) -> float:
    """Study-level fracture ROC AUC -- the metric to select models on.

    probs  : [N, C] softmax output
    labels : [N] integer class indices
    Returns 0.5 when the split contains only one class (AUC undefined).
    """
    from sklearn.metrics import roc_auc_score

    broken_idx = BROKEN_INDICES if broken_idx is None else list(broken_idx)
    is_broken = IS_BROKEN if is_broken is None else list(is_broken)

    img_scores = np.asarray(probs)[:, broken_idx].sum(axis=1)
    img_binary = np.array([is_broken[int(c)] for c in labels])

    _, s_score, s_true, _ = aggregate_studies(study_ids, img_scores, img_binary)
    if s_true is None or len(np.unique(s_true)) < 2:
        return 0.5
    return float(roc_auc_score(s_true, s_score))


# --------------------------------------------------------------------------
# Multi-head model
# --------------------------------------------------------------------------
class MultiHeadNet(nn.Module):
    """Shared backbone, separate body-part and fracture heads.

    WHY THIS EXISTS
    ---------------
    The 6-way formulation bundles two problems of wildly different difficulty
    into one softmax:

        body part  (Hand / Shoulder / Wrist)  -- trivially easy, ~99%
        fracture   (Broken / Healthy)         -- genuinely hard, ~0.88 AUC

    Two things go wrong as a result.

    1. The easy task dominates the gradient. Almost all of the cross-entropy
       loss is driven by body part, which the network solves in the first
       epoch; the fracture signal is a rounding error on top of it.

    2. Class weighting cannot be aimed. Inverse-frequency weights over six
       classes conflate *body-part* frequency with *fracture* frequency --
       upweighting `Hand_Broken` also upweights "hand-ness", which needed no
       help at all. train_v4.py's class-weighted loss has this problem.

    Splitting the heads fixes both: the two losses are independently weighted,
    and `pos_weight` on the fracture head corrects fracture imbalance alone.

    THE 6-WAY INTERFACE IS PRESERVED
    --------------------------------
    forward() returns log P over the same six classes in the same order, built
    as P(part) * P(condition). Because those six probabilities sum to one,
    softmax(log p) == p exactly -- so evaluate.py, predict.py and gradcam.py,
    which all call softmax on the model output, work against this model
    unchanged and produce genuine probabilities.

    P(fracture) also becomes a single calibrated sigmoid rather than a sum of
    three softmax entries, which is a better-behaved score to threshold.
    """

    def __init__(self, arch: str = "densenet121", pretrained: bool = False,
                 num_classes: int = NUM_CLASSES,
                 classes: Sequence[str] = CLASSES, dropout: float = 0.0):
        super().__init__()
        classes = list(classes)
        if len(classes) != num_classes:
            classes = list(CLASSES)
        if len(classes) != num_classes:
            raise ValueError(
                f"MultiHeadNet expects {len(CLASSES)} classes, got {num_classes}.")

        self.backbone_arch = arch.lower()
        self.classes = classes
        self.n_parts = len(BODY_PARTS)

        # Build the requested backbone, then strip its classifier so it emits a
        # pooled feature vector both heads can share.
        stub = build_model(self.backbone_arch, num_classes=1, pretrained=pretrained)
        if hasattr(stub, "classifier"):          # densenet
            feat_dim = stub.classifier.in_features
            stub.classifier = nn.Identity()
        elif hasattr(stub, "fc"):                # resnet
            feat_dim = stub.fc.in_features
            stub.fc = nn.Identity()
        else:
            raise ValueError(f"Don't know how to strip the head off {arch!r}.")

        self.backbone = stub
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.part_head = nn.Linear(feat_dim, self.n_parts)
        self.fx_head = nn.Linear(feat_dim, 1)

        # Label decomposition, as buffers so .to(device) moves them. Not
        # persistent: they are derived from `classes`, not learned, and saving
        # them would make old checkpoints fail the strict key check.
        self.register_buffer(
            "part_index",
            torch.tensor([BODY_PARTS.index(c.split("_")[0]) for c in classes],
                         dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "broken_index",
            torch.tensor([1 if c.endswith("_Broken") else 0 for c in classes],
                         dtype=torch.long),
            persistent=False)

    # -- the two heads, for training -------------------------------------
    def forward_heads(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (part_logits [B, 3], fracture_logit [B])."""
        feats = self.dropout(self.backbone(x))
        return self.part_head(feats), self.fx_head(feats).squeeze(-1)

    def fracture_logit(self, x) -> torch.Tensor:
        """Raw fracture logit. Grad-CAM backprops from this, not from argmax."""
        return self.forward_heads(x)[1]

    # -- the 6-way view, for everything else ------------------------------
    def forward(self, x) -> torch.Tensor:
        part_logits, fx_logit = self.forward_heads(x)
        log_p_part = torch.log_softmax(part_logits, dim=1)          # [B, 3]
        log_p_fx = nn.functional.logsigmoid(fx_logit)               # [B]
        log_p_ok = nn.functional.logsigmoid(-fx_logit)              # [B]
        log_p_cond = torch.stack([log_p_ok, log_p_fx], dim=1)       # [B, 2]
        # log P(class) = log P(part) + log P(condition); rows sum to 1 in
        # probability space, so a downstream softmax is the identity.
        return log_p_part[:, self.part_index] + log_p_cond[:, self.broken_index]


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int,
                    warmup_epochs: int = 1):
    """Linear warmup then cosine decay, stepped once per batch.

    Replaces ReduceLROnPlateau, which in v3 was tracking a different signal
    (validation loss) from the early-stopping criterion (validation accuracy),
    with different patience values. One schedule, one signal.
    """
    import math

    import torch.optim as optim

    total = max(epochs * steps_per_epoch, 1)
    warm = max(warmup_epochs * steps_per_epoch, 1)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(total - warm, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def is_multihead(model: nn.Module) -> bool:
    return isinstance(model, MultiHeadNet)


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """The last spatial layer worth visualising, per architecture.

    DenseNet's final BatchNorm and ResNet's last residual block are the
    conventional choices: the deepest feature map that still has spatial
    extent.
    """
    net = model.backbone if isinstance(model, MultiHeadNet) else model
    if hasattr(net, "features") and hasattr(net.features, "denseblock4"):
        # NOT norm5. torchvision's DenseNet.forward does
        #     out = F.relu(features, inplace=True)
        # on norm5's output. register_full_backward_hook wraps that output in a
        # custom autograd Function whose result is a view, and modifying a view
        # from a custom Function in place is forbidden -- autograd raises rather
        # than return wrong gradients. denseblock4 feeds norm5 (out-of-place
        # BatchNorm), so it is the deepest layer with spatial extent that
        # nothing mutates. Same 7x7 map at 224px.
        return net.features.denseblock4                # densenet121                # densenet121
    if hasattr(net, "layer4"):
        return net.layer4[-1]                          # resnet18 / resnet50
    raise ValueError(
        f"No known Grad-CAM target layer for {type(net).__name__}. "
        "Pass --layer explicitly.")
