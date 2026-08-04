"""
train_v4.py -- successor to Train_Full_Dataset.py.

What changed and why, in rough order of expected impact on your 0.885
study-level AUC:

  1. RESOLUTION 320, not 224.
     The MURA paper trains at 320x320. Fracture lines are thin, high-frequency
     features; downsampling to 224 throws away exactly the detail that carries
     the signal. This is the single biggest expected gain.

  2. CLASS-WEIGHTED LOSS.
     Your v3 evaluation showed the sensitivity/specificity gap tracking class
     imbalance almost perfectly by body part:
         Shoulder  49.4% fractures -> sens 0.799 / spec 0.814  (gap 0.015)
         Wrist     44.8% fractures -> sens 0.749 / spec 0.945  (gap 0.196)
         Hand      41.1% fractures -> sens 0.577 / spec 0.926  (gap 0.349)
     The model learned the prior, not just the pathology. Inverse-frequency
     weights should close most of that, especially on hand.

  3. MODEL SELECTION ON STUDY-LEVEL AUC, not image accuracy.
     Study-level fracture AUC is the metric you actually care about, so select
     and early-stop on it. v3 selected on 6-way image accuracy, which is
     dominated by the easy body-part sub-problem.

  4. DETERMINISTIC VALIDATION TRANSFORM. Already correct in v3; preserved here.

  5. SEEDED. v1/v2/v3 seeded nothing, so runs were not comparable.

  6. GradScaler WITH autocast. v3 used mixed-precision forward passes without
     loss scaling, risking gradient underflow in fp16.

  7. SELF-DESCRIBING CHECKPOINTS via common.save_checkpoint(), so the predict
     scripts can never again load DenseNet weights into a ResNet shell.

  8. COSINE LR SCHEDULE with warmup instead of ReduceLROnPlateau, and a single
     consistent signal for both scheduling and stopping.

Usage
-----
    python train_v4.py --train D:\\archive\\MURA-v1.1\\train \\
                       --valid D:\\archive\\MURA-v1.1\\valid \\
                       --out   D:\\archive\\Project_Dataset\\v4_densenet.pth

    # smaller/faster sanity run
    python train_v4.py --size 224 --epochs 3 --limit-batches 20
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
    MURADataset,
    NUM_CLASSES,
    build_model,
    build_scheduler,
    clean_path,
    eval_transform,
    pick_device,
    save_checkpoint,
    set_seed,
    study_level_auc,
    train_transform,
)

DEFAULT_TRAIN = r"D:\archive\MURA-v1.1\train"
DEFAULT_VALID = r"D:\archive\MURA-v1.1\valid"
DEFAULT_OUT = r"D:\archive\Project_Dataset\v4_densenet.pth"


# study_level_auc and build_scheduler used to be defined here (and
# study_level_auc a second time in evaluate.py, with nothing keeping the two
# implementations honest). Both now live in common.py, so the metric you select
# on is provably the metric you report.


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a MURA fracture classifier.")
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--valid", default=DEFAULT_VALID)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--arch", default="densenet121",
                    help="densenet121 / resnet18 / resnet50")
    ap.add_argument("--size", type=int, default=320,
                    help="Input resolution. 320 matches the MURA paper; 224 is "
                         "faster but loses fine fracture detail.")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Lower than v3's 32 because 320px uses ~2x the memory.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Disable inverse-frequency class weighting.")
    ap.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="Stop each epoch after N batches (smoke testing).")
    ap.add_argument("--log", default=None, help="Write per-epoch history to JSON.")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    use_amp = (not args.no_amp) and device.type in ("cuda", "xpu")

    print("=" * 70)
    print("  MURA fracture classifier -- v4")
    print("=" * 70)
    print(f"  device       {device}")
    print(f"  arch         {args.arch}")
    print(f"  resolution   {args.size}x{args.size}")
    print(f"  batch size   {args.batch_size}")
    print(f"  mixed prec.  {use_amp}")
    print(f"  seed         {args.seed}")
    print()

    # ---------------------------------------------------------------- data
    train_ds = MURADataset(clean_path(args.train), transform=train_transform(args.size))
    valid_ds = MURADataset(clean_path(args.valid), transform=eval_transform(args.size))

    print("\nTraining class distribution:")
    for name, count in train_ds.class_counts().items():
        print(f"  {name:<20s} {count:6d}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"),
                              drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # ---------------------------------------------------------------- model
    model = build_model(args.arch, NUM_CLASSES, pretrained=True).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("\nLoss: unweighted CrossEntropyLoss")
    else:
        weights = train_ds.class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("\nLoss: class-weighted CrossEntropyLoss")
        for name, w in zip(CLASSES, weights.cpu().numpy()):
            print(f"  {name:<20s} weight {w:.3f}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = len(train_loader) if not args.limit_batches else args.limit_batches
    scheduler = build_scheduler(optimizer, args.epochs, steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    broken_idx = [i for i, c in enumerate(CLASSES) if c.endswith("_Broken")]
    is_broken = [1 if c.endswith("_Broken") else 0 for c in CLASSES]

    best_auc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    best_epoch = -1
    no_improve = 0
    history = []
    start = time.time()

    # ---------------------------------------------------------------- loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        model.train()
        run_loss, run_correct, seen = 0.0, 0, 0

        for step, (images, labels, _) in enumerate(train_loader, 1):
            if args.limit_batches and step > args.limit_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            bs = images.size(0)
            run_loss += loss.item() * bs
            run_correct += int((outputs.argmax(1) == labels).sum().item())
            seen += bs

            if step % 50 == 0:
                print(f"  step {step}  loss {run_loss / seen:.4f}  "
                      f"acc {run_correct / seen:.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

        tr_loss = run_loss / max(seen, 1)
        tr_acc = run_correct / max(seen, 1)
        print(f"  TRAIN  loss {tr_loss:.4f}  acc {tr_acc:.4f}")

        # ------------------------------------------------------ validation
        model.eval()
        v_loss, v_correct, v_seen = 0.0, 0, 0
        all_probs, all_labels, all_idx = [], [], []

        with torch.no_grad():
            for step, (images, labels, idx) in enumerate(valid_loader, 1):
                if args.limit_batches and step > args.limit_batches:
                    break
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                bs = images.size(0)
                v_loss += loss.item() * bs
                v_correct += int((outputs.argmax(1) == labels).sum().item())
                v_seen += bs
                all_probs.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_idx.append(idx.numpy())

        probs = np.concatenate(all_probs)
        labels_np = np.concatenate(all_labels)
        idx_np = np.concatenate(all_idx)
        study_ids = [valid_ds.study_ids[i] for i in idx_np]

        val_loss = v_loss / max(v_seen, 1)
        val_acc = v_correct / max(v_seen, 1)
        val_auc = study_level_auc(probs, labels_np, study_ids, broken_idx, is_broken)

        print(f"  VALID  loss {val_loss:.4f}  6-way acc {val_acc:.4f}")
        print(f"         study-level fracture AUC {val_auc:.4f}   <- selection metric")

        history.append({"epoch": epoch + 1, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": val_loss,
                        "val_acc": val_acc, "val_study_auc": val_auc,
                        "lr": scheduler.get_last_lr()[0]})

        if val_auc > best_auc:
            best_auc = val_auc
            best_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
            print(f"  [*] New best study-level AUC. Checkpointing.")
            save_checkpoint(args.out, model, arch=args.arch, classes=CLASSES,
                            val_study_auc=val_auc, val_acc=val_acc,
                            epoch=epoch + 1, size=args.size, seed=args.seed)
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
    save_checkpoint(args.out, model, arch=args.arch, classes=CLASSES,
                    val_study_auc=best_auc, epoch=best_epoch,
                    size=args.size, seed=args.seed)

    print("\n" + "=" * 70)
    print(f"  Done in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"  Best study-level fracture AUC: {best_auc:.4f}  (epoch {best_epoch})")
    print(f"  Weights: {args.out}")
    print("=" * 70)
    print(f"\nNow evaluate properly:")
    print(f"  python evaluate.py --weights {args.out} --data {args.valid} "
          f"--size {args.size} --tta")

    if args.log:
        with open(args.log, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "history": history,
                       "best_auc": best_auc, "best_epoch": best_epoch}, fh, indent=2)
        print(f"  history -> {args.log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
