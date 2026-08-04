"""
train_v5.py -- multi-head successor to train_v4.py.

THE ONE IDEA
------------
Stop asking one softmax to solve two problems.

v1-v4 all trained a single 6-way classifier over {Hand,Shoulder,Wrist} x
{Broken,Healthy}. That formulation has two structural defects, and v4's
class-weighted loss only papers over the second:

  1. THE EASY TASK EATS THE GRADIENT.
     Body part is ~99% solvable and the network gets there in epoch one. After
     that, most of the remaining cross-entropy is still body-part loss noise,
     and the fracture signal -- the thing you actually care about -- is a small
     residual on top. You cannot turn the fracture task up without also turning
     the body-part task up.

  2. CLASS WEIGHTS CANNOT BE AIMED.
     v4 applies inverse-frequency weights over six classes. But a class's
     frequency is the product of how common its body part is and how common
     fractures are in that body part. Upweighting `Hand_Broken` therefore
     upweights "hand-ness" too -- a sub-problem that was already at 99% and
     needed no help. The correction leaks into the wrong task.

v5 uses a shared backbone with two heads: a 3-way body-part head and a single
binary fracture head. Now:

  * the two losses have independent weights (`--part-weight`, `--fx-weight`),
    so you can explicitly tell the optimiser the body-part task is nearly free;
  * `pos_weight` on the fracture head corrects fracture imbalance and NOTHING
    ELSE, which is the correction v4 was reaching for;
  * P(fracture) is one calibrated sigmoid instead of a sum of three softmax
    entries, so thresholds behave more predictably.

Everything downstream is unchanged. MultiHeadNet.forward() returns log P over
the same six classes in the same order, and because those probabilities sum to
one, the softmax that evaluate.py / predict.py / gradcam.py apply recovers them
exactly. Run the same evaluate.py against a v5 checkpoint and the numbers are
directly comparable to v4's.

Everything good about v4 is kept: 320px, seeded, GradScaler, cosine schedule,
deterministic validation transform, selection on study-level fracture AUC,
self-describing checkpoints.

Usage
-----
    python train_v5.py --train D:\\archive\\MURA-v1.1\\train \\
                       --valid D:\\archive\\MURA-v1.1\\valid \\
                       --out   D:\\archive\\Project_Dataset\\v5_multihead.pth

    # fast sanity check before committing a real run
    python train_v5.py --size 224 --epochs 2 --limit-batches 10

Then, as always:

    python evaluate.py --weights v5_multihead.pth --data ...\\valid --tta
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common import (
    CLASSES,
    IS_BROKEN,
    MULTIHEAD_PREFIX,
    MURADataset,
    MultiHeadNet,
    PART_OF,
    aggregate_studies,
    build_scheduler,
    clean_path,
    eval_transform,
    pick_device,
    save_checkpoint,
    set_seed,
    train_transform,
)

DEFAULT_TRAIN = r"D:\archive\MURA-v1.1\train"
DEFAULT_VALID = r"D:\archive\MURA-v1.1\valid"
DEFAULT_OUT = r"D:\archive\Project_Dataset\v5_multihead.pth"


# --------------------------------------------------------------------------
def split_labels(labels: torch.Tensor, part_of: torch.Tensor,
                 is_broken: torch.Tensor):
    """Decompose 6-way labels into (part_label [B], fracture_label [B] float)."""
    return part_of[labels], is_broken[labels].float()


def fracture_pos_weight(dataset: MURADataset) -> float:
    """n_healthy / n_broken -- the pos_weight for BCEWithLogitsLoss.

    This is the correction v4 was trying to make with 6-way class weights, but
    aimed at the fracture axis alone. If MURA is 58% healthy overall, this is
    about 1.4: each positive counts 1.4x as much as a negative, and body-part
    frequency does not enter into it at all.
    """
    broken = sum(IS_BROKEN[l] for l in dataset.labels)
    healthy = len(dataset.labels) - broken
    if broken == 0:
        return 1.0
    return float(healthy) / float(broken)


def evaluate_epoch(model, loader, dataset, device, part_of_t, is_broken_t,
                   criterion_part, criterion_fx, part_weight, fx_weight,
                   use_amp, limit_batches=0):
    """One validation pass. Returns a dict of metrics including the study AUC."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    tot_loss, seen = 0.0, 0
    part_correct, six_correct = 0, 0
    fx_scores, fx_true, all_idx = [], [], []

    with torch.no_grad():
        for step, (images, labels, idx) in enumerate(loader, 1):
            if limit_batches and step > limit_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            part_lab, fx_lab = split_labels(labels, part_of_t, is_broken_t)

            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    part_logits, fx_logit = model.forward_heads(images)
                    loss = (part_weight * criterion_part(part_logits, part_lab)
                            + fx_weight * criterion_fx(fx_logit.float(), fx_lab))
            else:
                part_logits, fx_logit = model.forward_heads(images)
                loss = (part_weight * criterion_part(part_logits, part_lab)
                        + fx_weight * criterion_fx(fx_logit, fx_lab))

            bs = images.size(0)
            tot_loss += float(loss.item()) * bs
            seen += bs

            part_pred = part_logits.argmax(1)
            part_correct += int((part_pred == part_lab).sum().item())
            fx_prob = torch.sigmoid(fx_logit.float())
            # 6-way prediction implied by the two heads, for comparability with
            # the accuracy v1-v4 printed.
            six_pred_part = part_pred
            six_pred_broken = (fx_prob >= 0.5).long()
            six_correct += int(((part_of_t[labels] == six_pred_part) &
                                (is_broken_t[labels] == six_pred_broken)).sum().item())

            fx_scores.append(fx_prob.cpu().numpy())
            fx_true.append(fx_lab.cpu().numpy())
            all_idx.append(idx.numpy())

    scores = np.concatenate(fx_scores)
    truth = np.concatenate(fx_true).astype(int)
    idx_np = np.concatenate(all_idx)
    study_ids = [dataset.study_ids[i] for i in idx_np]

    img_auc = (float(roc_auc_score(truth, scores))
               if len(np.unique(truth)) > 1 else 0.5)
    _, s_score, s_true, _ = aggregate_studies(study_ids, scores, truth)
    study_auc = (float(roc_auc_score(s_true, s_score))
                 if len(np.unique(s_true)) > 1 else 0.5)

    return {
        "loss": tot_loss / max(seen, 1),
        "part_acc": part_correct / max(seen, 1),
        "six_acc": six_correct / max(seen, 1),
        "image_auc": img_auc,
        "study_auc": study_auc,
        "n_studies": len(s_true),
    }


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train a multi-head MURA fracture classifier (v5).")
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--valid", default=DEFAULT_VALID)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--arch", default="densenet121",
                    help="Backbone: densenet121 / resnet18 / resnet50.")
    ap.add_argument("--size", type=int, default=320,
                    help="Input resolution. 320 matches the MURA paper.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=10.0,
                    help="Multiply the LR for the two heads. They start from "
                         "random init while the backbone starts from ImageNet, "
                         "so they need to move faster.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout on the shared feature vector before the heads.")
    ap.add_argument("--part-weight", type=float, default=0.2,
                    help="Weight on the body-part loss. Deliberately low: the "
                         "task is ~99%% solvable and does not need the capacity. "
                         "Keep it non-zero so the shared features stay "
                         "anatomically grounded.")
    ap.add_argument("--fx-weight", type=float, default=1.0,
                    help="Weight on the fracture loss -- the task you care about.")
    ap.add_argument("--no-pos-weight", action="store_true",
                    help="Disable fracture-class imbalance correction.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="Stop each epoch after N batches (smoke testing).")
    ap.add_argument("--log", default=None, help="Write per-epoch history to JSON.")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    use_amp = (not args.no_amp) and device.type in ("cuda", "xpu")

    print("=" * 72)
    print("  MURA fracture classifier -- v5 (multi-head)")
    print("=" * 72)
    print(f"  device        {device}")
    print(f"  backbone      {args.arch}")
    print(f"  resolution    {args.size}x{args.size}")
    print(f"  batch size    {args.batch_size}")
    print(f"  loss weights  part {args.part_weight}  fracture {args.fx_weight}")
    print(f"  mixed prec.   {use_amp}")
    print(f"  seed          {args.seed}")

    # ---------------------------------------------------------------- data
    train_ds = MURADataset(clean_path(args.train), transform=train_transform(args.size))
    valid_ds = MURADataset(clean_path(args.valid), transform=eval_transform(args.size))

    print("\nTraining class distribution:")
    for name, count in train_ds.class_counts().items():
        print(f"  {name:<20s} {count:6d}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers,
                              pin_memory=(device.type == "cuda"))

    # ---------------------------------------------------------------- model
    arch = MULTIHEAD_PREFIX + args.arch
    model = MultiHeadNet(args.arch, pretrained=True,
                         dropout=args.dropout).to(device)

    part_of_t = torch.tensor(PART_OF, dtype=torch.long, device=device)
    is_broken_t = torch.tensor(IS_BROKEN, dtype=torch.long, device=device)

    criterion_part = nn.CrossEntropyLoss()
    if args.no_pos_weight:
        criterion_fx = nn.BCEWithLogitsLoss()
        print("\nFracture loss: unweighted BCEWithLogitsLoss")
    else:
        pw = fracture_pos_weight(train_ds)
        criterion_fx = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pw, device=device))
        print(f"\nFracture loss: BCEWithLogitsLoss(pos_weight={pw:.3f})")
        print("  -- corrects fracture imbalance only, leaving body-part "
              "frequency alone.")

    # Heads start from scratch; the backbone starts from ImageNet.
    head_params = list(model.part_head.parameters()) + list(model.fx_head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": args.lr},
         {"params": head_params, "lr": args.lr * args.head_lr_mult}],
        weight_decay=args.weight_decay)

    steps = args.limit_batches or len(train_loader)
    scheduler = build_scheduler(optimizer, args.epochs, steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    best_auc, best_epoch, no_improve = 0.0, -1, 0
    best_wts = copy.deepcopy(model.state_dict())
    history = []
    start = time.time()

    # ---------------------------------------------------------------- loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 46)

        model.train()
        run_loss, run_part_loss, run_fx_loss = 0.0, 0.0, 0.0
        part_correct, fx_correct, seen = 0, 0, 0

        for step, (images, labels, _) in enumerate(train_loader, 1):
            if args.limit_batches and step > args.limit_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            part_lab, fx_lab = split_labels(labels, part_of_t, is_broken_t)

            optimizer.zero_grad(set_to_none=True)

            def compute():
                part_logits, fx_logit = model.forward_heads(images)
                l_part = criterion_part(part_logits, part_lab)
                l_fx = criterion_fx(fx_logit.float(), fx_lab)
                return part_logits, fx_logit, l_part, l_fx

            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    part_logits, fx_logit, l_part, l_fx = compute()
                    loss = args.part_weight * l_part + args.fx_weight * l_fx
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                part_logits, fx_logit, l_part, l_fx = compute()
                loss = args.part_weight * l_part + args.fx_weight * l_fx
                loss.backward()
                optimizer.step()

            scheduler.step()

            bs = images.size(0)
            run_loss += float(loss.item()) * bs
            run_part_loss += float(l_part.item()) * bs
            run_fx_loss += float(l_fx.item()) * bs
            part_correct += int((part_logits.argmax(1) == part_lab).sum().item())
            fx_correct += int(((torch.sigmoid(fx_logit.float()) >= 0.5).float()
                               == fx_lab).sum().item())
            seen += bs

            if step % 50 == 0:
                print(f"  step {step:4d}  loss {run_loss / seen:.4f}  "
                      f"(part {run_part_loss / seen:.4f} / fx {run_fx_loss / seen:.4f})  "
                      f"part-acc {part_correct / seen:.4f}  "
                      f"fx-acc {fx_correct / seen:.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

        tr = {"loss": run_loss / max(seen, 1),
              "part_loss": run_part_loss / max(seen, 1),
              "fx_loss": run_fx_loss / max(seen, 1),
              "part_acc": part_correct / max(seen, 1),
              "fx_acc": fx_correct / max(seen, 1)}
        print(f"  TRAIN  loss {tr['loss']:.4f}  part-acc {tr['part_acc']:.4f}  "
              f"fx-acc {tr['fx_acc']:.4f}")

        # ------------------------------------------------------ validation
        va = evaluate_epoch(model, valid_loader, valid_ds, device,
                            part_of_t, is_broken_t, criterion_part, criterion_fx,
                            args.part_weight, args.fx_weight, use_amp,
                            args.limit_batches)

        print(f"  VALID  loss {va['loss']:.4f}  6-way acc {va['six_acc']:.4f}  "
              f"body-part acc {va['part_acc']:.4f}")
        print(f"         image fracture AUC {va['image_auc']:.4f}")
        print(f"         STUDY fracture AUC {va['study_auc']:.4f}  "
              f"({va['n_studies']} studies)   <- selection metric")

        history.append({"epoch": epoch + 1, "train": tr, "valid": va,
                        "lr": scheduler.get_last_lr()[0]})

        if va["study_auc"] > best_auc:
            best_auc = va["study_auc"]
            best_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
            print("  [*] New best study-level AUC. Checkpointing.")
            save_checkpoint(args.out, model, arch=arch, classes=CLASSES,
                            val_study_auc=va["study_auc"],
                            val_image_auc=va["image_auc"],
                            val_six_acc=va["six_acc"],
                            val_part_acc=va["part_acc"],
                            epoch=epoch + 1, size=args.size, seed=args.seed,
                            part_weight=args.part_weight, fx_weight=args.fx_weight)
        else:
            no_improve += 1
            print(f"  [-] No improvement ({no_improve}/{args.patience}). "
                  f"Best {best_auc:.4f} @ epoch {best_epoch}")

        if no_improve >= args.patience:
            print("\n*** Early stopping ***")
            break

    # ---------------------------------------------------------------- done
    elapsed = time.time() - start
    model.load_state_dict(best_wts)
    save_checkpoint(args.out, model, arch=arch, classes=CLASSES,
                    val_study_auc=best_auc, epoch=best_epoch,
                    size=args.size, seed=args.seed,
                    part_weight=args.part_weight, fx_weight=args.fx_weight)

    print("\n" + "=" * 72)
    print(f"  Done in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"  Best study-level fracture AUC: {best_auc:.4f}  (epoch {best_epoch})")
    print(f"  Weights: {args.out}")
    print("=" * 72)
    print("\n  v3 (master_densenet_weights.pth) scored 0.885 study AUC / 0.649 kappa.")
    print("  Compare like for like with the same evaluate.py invocation:\n")
    print(f"    python evaluate.py --weights {args.out} --data {args.valid} --tta")

    if args.log:
        with open(args.log, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "history": history,
                       "best_study_auc": best_auc, "best_epoch": best_epoch},
                      fh, indent=2)
        print(f"\n  history -> {args.log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
