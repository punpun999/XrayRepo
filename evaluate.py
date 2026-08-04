"""
evaluate.py -- honest evaluation of a trained model on the official MURA
validation split.

Why this script exists
----------------------
The training scripts only ever printed a single 6-way accuracy number. That
number is misleading, because the 6-class problem bundles together:

  * body part   (Hand vs Shoulder vs Wrist) -- trivially easy, ~99% achievable
  * fracture    (Broken vs Healthy)         -- genuinely hard, published MURA
                                               baselines sit around 0.70 AUC

A model that nails body part and coin-flips on fractures still scores ~50%
overall and can look respectable. This script separates the two so you can see
what the network actually learned.

It reports, in increasing order of how much you should trust it:

  1. 6-way accuracy + confusion matrix       (the number your training loop printed)
  2. Body-part-only accuracy                 (the easy sub-problem)
  3. Image-level fracture metrics            (the hard sub-problem: AUC, kappa,
                                              sensitivity, specificity)
  4. Per-body-part fracture breakdown        (wrists are usually easier than shoulders)
  5. Study-level fracture metrics            (MURA's official protocol -- the only
                                              numbers comparable to the literature)

Usage
-----
    python evaluate.py --weights D:\\archive\\Project_Dataset\\master_densenet_weights.pth \\
                       --data D:\\archive\\MURA-v1.1\\valid \\
                       --outdir eval_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    BODY_PARTS,
    MURADataset,
    aggregate_studies,
    CLASSES,
    clean_path,
    eval_transform,
    load_checkpoint,
    pick_device,
    set_seed,
)

DEFAULT_WEIGHTS = r"D:\archive\Project_Dataset\master_densenet_weights.pth"
DEFAULT_DATA = r"D:\archive\MURA-v1.1\valid"


# --------------------------------------------------------------------------
# Metric helpers (implemented directly so sklearn stays the only hard dep)
# --------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, y_score: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Full battery of binary-classification metrics.

    y_true  : 1 = fracture, 0 = healthy
    y_score : P(fracture)
    """
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    # AUC is undefined if only one class is present.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising Youden's J (sensitivity + specificity - 1).

    0.5 is arbitrary and usually wrong on an imbalanced dataset. Note that
    Youden weights sensitivity and specificity EQUALLY, which is the wrong
    objective for fracture screening -- see threshold_at_sensitivity().
    """
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return _clip_threshold(thr[int(np.argmax(j))])


def _clip_threshold(t) -> float:
    """roc_curve's first threshold is max(score)+1 (or inf). Keep it usable."""
    t = float(t)
    if not np.isfinite(t):
        return 1.0
    return float(min(max(t, 0.0), 1.0))


def threshold_at_sensitivity(y_true: np.ndarray, y_score: np.ndarray,
                             target: float = 0.85) -> float:
    """Lowest threshold that achieves at least `target` sensitivity.

    This is the clinically meaningful way to pick an operating point. A missed
    fracture (false negative) sends a patient home injured; a false positive
    costs a second look. The two errors are not symmetric, so Youden's J --
    which treats them as if they were -- is the wrong criterion.
    """
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return _clip_threshold(np.min(thr))
    return _clip_threshold(thr[int(ok[0])])


def operating_point_table(y_true: np.ndarray, y_score: np.ndarray,
                          targets=(0.80, 0.85, 0.90, 0.95)) -> list:
    """What specificity costs you at each target sensitivity."""
    rows = []
    for t in targets:
        thr = threshold_at_sensitivity(y_true, y_score, t)
        m = binary_metrics(y_true, y_score, threshold=thr)
        rows.append({
            "target_sensitivity": t,
            "threshold": thr,
            "sensitivity": m["recall_sensitivity"],
            "specificity": m["specificity"],
            "precision_ppv": m["precision_ppv"],
            "accuracy": m["accuracy"],
            "cohen_kappa": m["cohen_kappa"],
            "false_negatives": m["confusion"]["fn"],
            "false_positives": m["confusion"]["fp"],
        })
    return rows


def print_operating_points(rows, label: str) -> None:
    print(f"\n  Operating points -- {label}:")
    print(f"  {'target sens':>12s} {'thresh':>8s} {'sens':>7s} {'spec':>7s} "
          f"{'PPV':>7s} {'kappa':>7s} {'missed':>8s} {'false+':>8s}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['target_sensitivity']:>12.2f} {r['threshold']:>8.3f} "
              f"{r['sensitivity']:>7.3f} {r['specificity']:>7.3f} "
              f"{r['precision_ppv']:>7.3f} {r['cohen_kappa']:>7.3f} "
              f"{r['false_negatives']:>8d} {r['false_positives']:>8d}")


def print_confusion(cm: np.ndarray, labels) -> None:
    width = max(len(l) for l in labels) + 2
    header = " " * width + "".join(f"{l[:10]:>12s}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        cells = "".join(f"{v:>12d}" for v in row)
        print(f"{labels[i]:<{width}s}{cells}")


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


# --------------------------------------------------------------------------
# Inference over the dataset
# --------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, loader, device, dataset, amp: bool = False,
                  tta: bool = False):
    """Return (probs [N, C], labels [N], study_ids [N], order [N]).

    With tta=True, averages the softmax over the image and its horizontal
    mirror. MURA contains both left and right limbs, so laterality carries no
    diagnostic signal and the flip is a label-preserving transform. Costs one
    extra forward pass; typically worth a point or two of AUC for free.
    """
    model.eval()
    all_probs, all_labels, all_idx = [], [], []

    def forward(x):
        if amp and device.type in ("cuda", "xpu"):
            with torch.amp.autocast(device_type=device.type):
                return torch.softmax(model(x).float(), dim=1)
        return torch.softmax(model(x).float(), dim=1)

    total = len(loader)
    for step, (images, labels, idx) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        probs = forward(images)
        if tta:
            flipped = torch.flip(images, dims=[3])
            probs = (probs + forward(flipped)) / 2.0
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_idx.append(idx.numpy())
        if step % 20 == 0 or step == total:
            print(f"\r  batch {step}/{total}", end="", flush=True)
    print()

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    idx = np.concatenate(all_idx)
    # Index back through `idx` rather than assuming the loader preserved order.
    study_ids = np.array([dataset.study_ids[i] for i in idx])
    return probs, labels, study_ids, idx


# --------------------------------------------------------------------------
# Plotting (optional -- skipped cleanly if matplotlib is unavailable)
# --------------------------------------------------------------------------
def save_plots(cm6, y_true_bin, y_score_bin, classes, outdir) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[i] matplotlib not installed -- skipping plots.")
        return []

    from sklearn.metrics import roc_curve, precision_recall_curve

    written = []

    # --- 6-class confusion matrix, row-normalised ---
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = cm6.astype(float) / np.maximum(cm6.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm6[i, j]})",
                    ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = os.path.join(outdir, "confusion_matrix.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # --- ROC + PR for the binary fracture task ---
    if len(np.unique(y_true_bin)) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        fpr, tpr, _ = roc_curve(y_true_bin, y_score_bin)
        from sklearn.metrics import roc_auc_score, average_precision_score
        axes[0].plot(fpr, tpr, lw=2,
                     label=f"AUC = {roc_auc_score(y_true_bin, y_score_bin):.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].set_title("ROC -- fracture vs healthy")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)

        prec, rec, _ = precision_recall_curve(y_true_bin, y_score_bin)
        axes[1].plot(rec, prec, lw=2,
                     label=f"AP = {average_precision_score(y_true_bin, y_score_bin):.3f}")
        axes[1].axhline(y_true_bin.mean(), ls="--", c="k", lw=1,
                        label=f"prevalence = {y_true_bin.mean():.3f}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall -- fracture vs healthy")
        axes[1].legend(loc="lower left")
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        p = os.path.join(outdir, "fracture_curves.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    return written


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate a MURA model with metrics that separate the easy "
                    "(body part) sub-problem from the hard (fracture) one.")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="MURA valid/test directory (NOT the sampled Train_Data folder).")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. Keep at 0 on Windows if you hit issues.")
    ap.add_argument("--size", type=int, default=None,
                    help="Input resolution. Defaults to whatever the checkpoint "
                         "was trained at, or 224 for legacy checkpoints.")
    ap.add_argument("--amp", action="store_true", help="Mixed precision inference.")
    ap.add_argument("--tta", action="store_true",
                    help="Test-time augmentation: average over the image and its "
                         "horizontal mirror. Doubles inference time, usually "
                         "worth 1-2 points of AUC for free.")
    ap.add_argument("--target-sensitivity", type=float, default=0.85,
                    help="Report an operating point achieving at least this "
                         "sensitivity. Fracture screening should favour "
                         "sensitivity over specificity. Default 0.85.")
    ap.add_argument("--outdir", default="eval_results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = pick_device(args.device)
    print(f"Device: {device}")

    try:
        ckpt = load_checkpoint(clean_path(args.weights), device=device, arch=args.arch)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[!] Could not load model: {exc}", file=sys.stderr)
        return 1
    size = args.size or ckpt.size or 224
    print(f"Model:  {ckpt.arch} ({ckpt.num_classes} classes) @ {size}px")
    if args.size and ckpt.size and args.size != ckpt.size:
        print(f"[!] WARNING: evaluating at {args.size}px but this checkpoint was "
              f"trained at {ckpt.size}px. Metrics will be depressed.")

    classes = list(ckpt.classes)
    if classes != CLASSES:
        print(f"[i] Checkpoint class order differs from default: {classes}")

    # Derive the label decompositions from the checkpoint's own class list so
    # this still works if a future model uses a different ordering.
    part_of = [BODY_PARTS.index(c.split("_")[0]) for c in classes]
    is_broken = [1 if c.endswith("_Broken") else 0 for c in classes]
    broken_idx = [i for i, b in enumerate(is_broken) if b == 1]

    dataset = MURADataset(clean_path(args.data), transform=eval_transform(size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    print("\nClass distribution in this split:")
    for name, count in dataset.class_counts().items():
        print(f"  {name:<20s} {count:6d}")

    print("\nRunning inference..." + ("  (with horizontal-flip TTA)" if args.tta else ""))
    probs, y_true, study_ids, order = run_inference(ckpt.model, loader, device,
                                                    dataset, amp=args.amp,
                                                    tta=args.tta)
    y_pred = probs.argmax(axis=1)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
    )

    results = {
        "weights": args.weights,
        "data": args.data,
        "arch": ckpt.arch,
        "n_images": int(len(y_true)),
        "n_studies": int(len(set(study_ids.tolist()))),
    }

    # ---------------------------------------------------------------- 1
    section("1. SIX-WAY CLASSIFICATION  (the number your training loop printed)")
    acc6 = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy: {acc6 * 100:.2f}%")
    print(f"Cohen's kappa:    {cohen_kappa_score(y_true, y_pred):.4f}")
    print("\nPer-class breakdown:")
    print(classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    cm6 = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    print("Confusion matrix (rows = true, cols = predicted):")
    print_confusion(cm6, classes)
    results["six_way"] = {
        "accuracy": float(acc6),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": cm6.tolist(),
        "labels": list(classes),
        "report": classification_report(
            y_true, y_pred, labels=list(range(len(classes))),
            target_names=classes, output_dict=True, zero_division=0),
    }

    # ---------------------------------------------------------------- 2
    section("2. BODY PART ONLY  (the easy sub-problem)")
    part_true = np.array([part_of[c] for c in y_true])
    part_pred = np.array([part_of[c] for c in y_pred])
    part_acc = accuracy_score(part_true, part_pred)
    print(f"\nBody-part accuracy: {part_acc * 100:.2f}%")
    print("(Expect ~99%. If the 6-way score is high but section 3 is near "
          "chance,\n this is where the accuracy is coming from.)")
    cm_part = confusion_matrix(part_true, part_pred, labels=[0, 1, 2])
    print()
    print_confusion(cm_part, BODY_PARTS)
    results["body_part"] = {
        "accuracy": float(part_acc),
        "confusion_matrix": cm_part.tolist(),
        "labels": BODY_PARTS,
    }

    # ---------------------------------------------------------------- 3
    section("3. FRACTURE DETECTION, IMAGE LEVEL  (the hard sub-problem)")
    y_true_bin = np.array([is_broken[c] for c in y_true])
    y_score_bin = probs[:, broken_idx].sum(axis=1)

    m = binary_metrics(y_true_bin, y_score_bin, threshold=0.5)
    thr = best_threshold(y_true_bin, y_score_bin)
    m_opt = binary_metrics(y_true_bin, y_score_bin, threshold=thr)

    print(f"\nPrevalence of fracture: {m['prevalence'] * 100:.1f}%")
    print(f"ROC AUC:                {m['roc_auc']:.4f}"
          if m["roc_auc"] is not None else "ROC AUC: n/a")
    print(f"PR  AUC:                {m['pr_auc']:.4f}"
          if m["pr_auc"] is not None else "")
    print("\n              threshold=0.50   threshold=%.3f (Youden-optimal)" % thr)
    for key, label in [("accuracy", "Accuracy"),
                       ("cohen_kappa", "Cohen's kappa"),
                       ("recall_sensitivity", "Sensitivity"),
                       ("specificity", "Specificity"),
                       ("precision_ppv", "Precision/PPV"),
                       ("f1", "F1")]:
        print(f"  {label:<16s} {m[key]:>10.4f}    {m_opt[key]:>10.4f}")
    c = m["confusion"]
    print(f"\n  At 0.50 -> TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")
    print(f"            {c['fn']} missed fractures vs {c['fp']} false alarms.")

    img_ops = operating_point_table(y_true_bin, y_score_bin)
    print_operating_points(img_ops, "image level")

    tgt_thr = threshold_at_sensitivity(y_true_bin, y_score_bin, args.target_sensitivity)
    m_tgt = binary_metrics(y_true_bin, y_score_bin, threshold=tgt_thr)
    print(f"\n  At your --target-sensitivity {args.target_sensitivity:.2f} "
          f"(threshold {tgt_thr:.3f}):")
    print(f"    sensitivity {m_tgt['recall_sensitivity']:.3f}   "
          f"specificity {m_tgt['specificity']:.3f}   "
          f"missed {m_tgt['confusion']['fn']} (was {c['fn']})")

    print("\nReference -- MURA paper, 169-layer DenseNet baseline:")
    print("  AUROC 0.929, operating point 0.815 sensitivity / 0.887 specificity.")
    print("  Cohen's kappa 0.705 on the radiologist-adjudicated test set")
    print("  (best individual radiologist: 0.778).")
    print("  Note that baseline trained on all 7 study types, not 3, and its")
    print("  kappa was scored against a different gold standard -- so treat the")
    print("  comparison as a rough reference, not a like-for-like benchmark.")
    results["fracture_image_level"] = {"at_0.5": m, "at_optimal": m_opt,
                                       "optimal_threshold": thr,
                                       "operating_points": img_ops,
                                       "at_target_sensitivity": m_tgt}

    # ---------------------------------------------------------------- 4
    section("4. FRACTURE DETECTION BY BODY PART")
    per_part = {}
    print()
    print(f"  {'Part':<10s} {'n':>6s} {'AUC':>8s} {'kappa':>8s} "
          f"{'sens':>8s} {'spec':>8s} {'acc':>8s}")
    print("  " + "-" * 56)
    for pi, part in enumerate(BODY_PARTS):
        mask = part_true == pi
        if mask.sum() == 0:
            continue
        pm = binary_metrics(y_true_bin[mask], y_score_bin[mask], threshold=0.5)
        per_part[part] = pm
        auc_s = f"{pm['roc_auc']:.4f}" if pm["roc_auc"] is not None else "   n/a"
        print(f"  {part:<10s} {pm['n']:>6d} {auc_s:>8s} {pm['cohen_kappa']:>8.4f} "
              f"{pm['recall_sensitivity']:>8.4f} {pm['specificity']:>8.4f} "
              f"{pm['accuracy']:>8.4f}")
    results["fracture_by_part"] = per_part

    # ---------------------------------------------------------------- 5
    section("5. FRACTURE DETECTION, STUDY LEVEL  (MURA's official protocol)")
    print("\nMURA labels studies, not images: a study is one patient's set of views\n"
          "of one limb. Averaging predictions across a study's images is the\n"
          "protocol used by published results, so these are the comparable numbers.")

    # Shared with train_v4/v5's selection metric, so the number you early-stop
    # on and the number you report here cannot drift apart.
    sids, s_score, s_true, s_part = aggregate_studies(
        study_ids, y_score_bin, labels=y_true_bin, parts=part_true)

    sm = binary_metrics(s_true, s_score, threshold=0.5)
    s_thr = best_threshold(s_true, s_score)
    sm_opt = binary_metrics(s_true, s_score, threshold=s_thr)

    print(f"\nStudies: {sm['n']}   fracture prevalence: {sm['prevalence'] * 100:.1f}%")
    if sm["roc_auc"] is not None:
        print(f"ROC AUC: {sm['roc_auc']:.4f}")
    print("\n              threshold=0.50   threshold=%.3f (Youden-optimal)" % s_thr)
    for key, label in [("accuracy", "Accuracy"),
                       ("cohen_kappa", "Cohen's kappa"),
                       ("recall_sensitivity", "Sensitivity"),
                       ("specificity", "Specificity"),
                       ("f1", "F1")]:
        print(f"  {label:<16s} {sm[key]:>10.4f}    {sm_opt[key]:>10.4f}")

    study_ops = operating_point_table(s_true, s_score)
    print_operating_points(study_ops, "study level")

    s_tgt_thr = threshold_at_sensitivity(s_true, s_score, args.target_sensitivity)
    s_tgt = binary_metrics(s_true, s_score, threshold=s_tgt_thr)
    print(f"\n  RECOMMENDED OPERATING POINT (target sensitivity "
          f"{args.target_sensitivity:.2f}):")
    print(f"    threshold   {s_tgt_thr:.4f}")
    print(f"    sensitivity {s_tgt['recall_sensitivity']:.4f}   "
          f"specificity {s_tgt['specificity']:.4f}")
    print(f"    missed fractures {s_tgt['confusion']['fn']} "
          f"(vs {sm['confusion']['fn']} at threshold 0.50)")

    print("\n  Per body part (study level):")
    print(f"  {'Part':<10s} {'n':>6s} {'AUC':>8s} {'kappa':>8s}")
    print("  " + "-" * 34)
    study_per_part = {}
    for pi, part in enumerate(BODY_PARTS):
        mask = s_part == pi
        if mask.sum() == 0:
            continue
        pm = binary_metrics(s_true[mask], s_score[mask], threshold=0.5)
        study_per_part[part] = pm
        auc_s = f"{pm['roc_auc']:.4f}" if pm["roc_auc"] is not None else "   n/a"
        print(f"  {part:<10s} {pm['n']:>6d} {auc_s:>8s} {pm['cohen_kappa']:>8.4f}")

    results["fracture_study_level"] = {
        "at_0.5": sm, "at_optimal": sm_opt, "optimal_threshold": s_thr,
        "operating_points": study_ops,
        "target_sensitivity": args.target_sensitivity,
        "recommended_threshold": s_tgt_thr,
        "at_recommended": s_tgt,
        "by_part": study_per_part,
    }
    results["tta"] = bool(args.tta)

    # ---------------------------------------------------------------- output
    section("OUTPUT")
    plots = save_plots(cm6, y_true_bin, y_score_bin, classes, args.outdir)

    json_path = os.path.join(args.outdir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    preds_path = os.path.join(args.outdir, "predictions.csv")
    with open(preds_path, "w", encoding="utf-8") as fh:
        fh.write("path,study_id,true_class,pred_class,p_fracture,"
                 + ",".join(f"p_{c}" for c in classes) + "\n")
        for i in range(len(y_true)):
            row = ",".join(f"{p:.6f}" for p in probs[i])
            fh.write(f'"{dataset.image_paths[order[i]]}","{study_ids[i]}",'
                     f"{classes[y_true[i]]},{classes[y_pred[i]]},"
                     f"{y_score_bin[i]:.6f},{row}\n")

    print(f"\n  metrics.json     -> {json_path}")
    print(f"  predictions.csv  -> {preds_path}")
    for p in plots:
        print(f"  plot             -> {p}")

    # ---------------------------------------------------------------- verdict
    section("VERDICT")
    k = results["fracture_study_level"]["at_0.5"]["cohen_kappa"]
    auc = results["fracture_study_level"]["at_0.5"]["roc_auc"] or 0.0
    print(f"\n  6-way accuracy:            {acc6 * 100:.2f}%")
    print(f"  Body-part accuracy:        {part_acc * 100:.2f}%   (easy)")
    print(f"  Study-level fracture AUC:  {auc:.4f}      (hard -- this is the real score)")
    print(f"  Study-level kappa:         {k:.4f}")
    if k < 0.10:
        print("\n  => The model has essentially NOT learned to detect fractures.\n"
              "     Its 6-way accuracy comes almost entirely from body part.")
    elif k < 0.40:
        print("\n  => Weak but non-random fracture signal. Below published baselines.")
    elif k < 0.65:
        print("\n  => Moderate agreement. In the neighbourhood of a reasonable baseline.")
    else:
        print("\n  => Strong agreement, at or above typical published MURA baselines.\n"
              "     Double-check for patient leakage between train and this split.")

    # Operating-point advice: is the model trading away sensitivity?
    sens = sm["recall_sensitivity"]
    spec = sm["specificity"]
    if spec - sens > 0.10:
        print(f"\n  [!] Operating point is skewed toward specificity "
              f"(sens {sens:.3f} vs spec {spec:.3f}).")
        print(f"      For fracture screening this is the wrong direction: you are")
        print(f"      missing {sm['confusion']['fn']} fractures to avoid "
              f"{sm['confusion']['fp']} false alarms.")
        print(f"      Use threshold {s_tgt_thr:.3f} instead -> sensitivity "
              f"{s_tgt['recall_sensitivity']:.3f}, specificity "
              f"{s_tgt['specificity']:.3f}.")
        print(f"      Pass that to predict.py with --threshold {s_tgt_thr:.3f}.")

    if not args.tta:
        print("\n  [i] Re-run with --tta for a free accuracy bump "
              "(averages over the horizontal mirror).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
