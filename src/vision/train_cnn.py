"""Train ScratchCNN baseline and two-phase Transfer-Learned ResNet18 (Area A)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.vision.dataset import get_dataloaders
from src.vision.models import ScratchCNN, build_transfer_resnet18, set_transfer_stage


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * images.size(0)
            _, top5_preds = logits.topk(5, dim=1)
            top1_correct += int((top5_preds[:, 0] == labels).sum().item())
            top5_correct += int((top5_preds == labels[:, None]).any(dim=1).sum().item())
            total += images.size(0)

    return total_loss / total, top1_correct / total, top5_correct / total


def train_one_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epochs: int,
    device: torch.device,
    save_path: Path,
    tag: str,
) -> Tuple[List[Dict], float]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_val_acc = 0.0
    if save_path.exists():
        # Preserve best accuracy across two-phase fine-tuning if checkpoint already exists
        _, best_val_acc, _ = evaluate_epoch(model, val_loader, criterion, device)

    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"[{tag}] Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += images.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_top1, val_top5 = evaluate_epoch(model, val_loader, criterion, device)
        dt = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_top1, 4),
            "val_top5": round(val_top5, 4),
            "seconds": round(dt, 1),
        }
        history.append(record)
        print(
            f"[{tag}] Epoch {epoch}/{epochs} | "
            f"train_acc={train_acc:.3f} val_top1={val_top1:.3f} val_top5={val_top5:.3f} ({dt:.1f}s)"
        )

        if val_top1 >= best_val_acc:
            best_val_acc = val_top1
            torch.save(model.state_dict(), save_path)

    return history, best_val_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/food101_10class")
    parser.add_argument("--out-dir", default="weights/vision")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scratch-epochs", type=int, default=12)
    parser.add_argument("--extract-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training vision models on device: {device}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    # 1. Scratch CNN
    scratch = ScratchCNN(num_classes=len(classes), dropout=0.4).to(device)
    scratch_opt = torch.optim.AdamW(scratch.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scratch_sched = torch.optim.lr_scheduler.CosineAnnealingLR(scratch_opt, T_max=args.scratch_epochs)
    scratch_path = out_dir / "scratch_cnn.pt"
    scratch_hist, _ = train_one_phase(
        scratch, train_loader, val_loader, scratch_opt, scratch_sched,
        args.scratch_epochs, device, scratch_path, tag="ScratchCNN"
    )

    # 2. Transfer-Learned ResNet18 (Phase 1: Feature Extraction, Phase 2: Fine-Tuning)
    resnet = build_transfer_resnet18(num_classes=len(classes), dropout=0.4, pretrained=True).to(device)
    resnet_path = out_dir / "resnet18_finetuned.pt"

    set_transfer_stage(resnet, "feature_extract")
    opt_p1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3, weight_decay=1e-4)
    p1_hist, _ = train_one_phase(
        resnet, train_loader, val_loader, opt_p1, None,
        args.extract_epochs, device, resnet_path, tag="ResNet18-Extract"
    )

    # Load best Phase 1 weights before Phase 2 fine-tuning
    resnet.load_state_dict(torch.load(resnet_path, map_location=device, weights_only=True))
    set_transfer_stage(resnet, "fine_tune")
    opt_p2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, resnet.parameters()), lr=2e-4, weight_decay=1e-4)
    sched_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=args.finetune_epochs)
    p2_hist, _ = train_one_phase(
        resnet, train_loader, val_loader, opt_p2, sched_p2,
        args.finetune_epochs, device, resnet_path, tag="ResNet18-FineTune"
    )

    # Quick held-out test check
    criterion = nn.CrossEntropyLoss()
    scratch.load_state_dict(torch.load(scratch_path, map_location=device, weights_only=True))
    _, s_top1, s_top5 = evaluate_epoch(scratch, test_loader, criterion, device)

    resnet.load_state_dict(torch.load(resnet_path, map_location=device, weights_only=True))
    _, r_top1, r_top5 = evaluate_epoch(resnet, test_loader, criterion, device)

    summary = {
        "classes": classes,
        "scratch_history": scratch_hist,
        "resnet_extract_history": p1_hist,
        "resnet_finetune_history": p2_hist,
        "test_preview": {
            "scratch_cnn": {"top1": round(s_top1, 4), "top5": round(s_top5, 4)},
            "resnet18_finetuned": {"top1": round(r_top1, 4), "top5": round(r_top5, 4)},
        },
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Vision training complete:", json.dumps(summary["test_preview"], indent=2))


if __name__ == "__main__":
    main()
