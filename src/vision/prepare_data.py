"""Download Food-101 via Hugging Face Hub (`ethz/food101`) and extract a 10-class train/val/test subset."""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

from src.vision.dataset import SELECTED_CLASSES


def _extract_label_names(pf: pq.ParquetFile) -> list[str]:
    meta = pf.schema_arrow.metadata or {}
    hf_raw = meta.get(b"huggingface")
    if hf_raw:
        info = json.loads(hf_raw.decode("utf-8"))
        return info["info"]["features"]["label"]["names"]
    raise RuntimeError("Could not find Hugging Face label names in parquet metadata.")


def prepare_food101_subset(
    raw_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/food101_10class"),
    samples_dir: Path = Path("data/samples"),
    val_ratio: float = 0.2,
    seed: int = 42,
    cleanup_raw: bool = False,
) -> None:
    if (out_dir / "test").exists() and len(list((out_dir / "test").glob("*/*.jpg"))) >= 2500:
        print(f"Prepared dataset already exists at {out_dir}")
        return

    if cleanup_raw and raw_dir.exists():
        shutil.rmtree(raw_dir, ignore_errors=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for cls_name in SELECTED_CLASSES:
            (out_dir / split / cls_name).mkdir(parents=True, exist_ok=True)

    print("Listing ethz/food101 parquet shards on Hugging Face Hub...")
    repo_files = list_repo_files("ethz/food101", repo_type="dataset")
    train_shards = sorted(f for f in repo_files if f.startswith("data/train-") and f.endswith(".parquet"))
    val_shards = sorted(f for f in repo_files if f.startswith("data/validation-") and f.endswith(".parquet"))

    target_set = set(SELECTED_CLASSES)
    label_names: list[str] | None = None
    target_ids: dict[int, str] = {}

    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "test": 0}
    train_buffers: dict[str, list[tuple[str, bytes]]] = {c: [] for c in SELECTED_CLASSES}

    # 1. Process official Food-101 validation (= held-out test) shards
    for shard in val_shards:
        print(f"Downloading/reading {shard}...")
        local_pq = hf_hub_download("ethz/food101", shard, repo_type="dataset")
        pf = pq.ParquetFile(local_pq)
        if label_names is None:
            label_names = _extract_label_names(pf)
            target_ids = {i: name for i, name in enumerate(label_names) if name in target_set}

        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["image", "label"])
            labels = table["label"].to_pylist()
            images = table["image"].to_pylist()
            for idx, (lbl, img_dict) in enumerate(zip(labels, images)):
                if lbl not in target_ids:
                    continue
                cls_name = target_ids[lbl]
                fname = Path(img_dict.get("path") or f"{cls_name}_{rg_idx}_{idx}.jpg").name
                if not fname.lower().endswith(".jpg"):
                    fname = f"{Path(fname).stem}.jpg"
                dest = out_dir / "test" / cls_name / fname
                if not dest.exists():
                    Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB").save(dest, format="JPEG", quality=92)
                counts["test"] += 1

    # 2. Process official Food-101 train shards, then split 80/20 into train/val
    for shard in train_shards:
        print(f"Downloading/reading {shard}...")
        local_pq = hf_hub_download("ethz/food101", shard, repo_type="dataset")
        pf = pq.ParquetFile(local_pq)
        if label_names is None:
            label_names = _extract_label_names(pf)
            target_ids = {i: name for i, name in enumerate(label_names) if name in target_set}

        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["image", "label"])
            labels = table["label"].to_pylist()
            images = table["image"].to_pylist()
            for idx, (lbl, img_dict) in enumerate(zip(labels, images)):
                if lbl not in target_ids:
                    continue
                cls_name = target_ids[lbl]
                fname = Path(img_dict.get("path") or f"{cls_name}_{rg_idx}_{idx}.jpg").name
                if not fname.lower().endswith(".jpg"):
                    fname = f"{Path(fname).stem}.jpg"
                train_buffers[cls_name].append((fname, img_dict["bytes"]))

    for cls_name in SELECTED_CLASSES:
        items = train_buffers[cls_name]
        rng.shuffle(items)
        n_val = int(len(items) * val_ratio)
        for split, subset in (("val", items[:n_val]), ("train", items[n_val:])):
            for fname, raw_bytes in subset:
                dest = out_dir / split / cls_name / fname
                if not dest.exists():
                    Image.open(io.BytesIO(raw_bytes)).convert("RGB").save(dest, format="JPEG", quality=92)
                counts[split] += 1

        # Copy 1 clean test image per class into data/samples/
        test_imgs = sorted((out_dir / "test" / cls_name).glob("*.jpg"))
        if test_imgs:
            shutil.copy2(test_imgs[0], samples_dir / f"{cls_name}.jpg")

    stats = {
        "source": "ETHZ Food-101 (ethz/food101 via Hugging Face Hub)",
        "classes": SELECTED_CLASSES,
        "counts": counts,
    }
    (out_dir / "metadata.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Prepared 10-class Food-101 subset:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-raw", action="store_true")
    args = parser.parse_args()
    prepare_food101_subset(cleanup_raw=args.cleanup_raw)
