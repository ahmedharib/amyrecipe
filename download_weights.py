"""Download or upload trained/quantized model weights from Hugging Face Hub (R7)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DEFAULT_HF_REPO = os.environ.get("SWE015_HF_REPO", "Ahmed-Harib/food-recipe-finder-weights")

REQUIRED_WEIGHT_FILES = [
    "weights/vision/classes.json",
    "weights/vision/scratch_cnn.pt",
    "weights/vision/resnet18_finetuned.pt",
    "weights/vision/resnet18_int8.onnx",
    "weights/neural_retriever/vocab.json",
    "weights/neural_retriever/neural_retriever.pt",
    "weights/neural_retriever/recipe_embeddings.npy",
    "weights/bert_retriever/config.json",
    "weights/bert_retriever/bert_int8.onnx",
    "weights/bert_retriever/recipe_embeddings.npy",
]


def weights_present(root: Path = Path(".")) -> bool:
    return all((root / rel).exists() for rel in REQUIRED_WEIGHT_FILES)


def ensure_weights(repo_id: str = DEFAULT_HF_REPO, root: Path = Path(".")) -> bool:
    if weights_present(root):
        print("[weights] All required FP32 & INT8 ONNX model weights are present locally.")
        return True

    print(f"[weights] Local weights missing; downloading from Hugging Face Hub ({repo_id})...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(root / "weights"),
        )
        print("[weights] Successfully downloaded weights into ./weights/")
        return True
    except Exception as exc:
        print(f"[weights] Could not download from {repo_id}: {exc}")
        print(
            "[weights] Tip: Upload your local ./weights folder with:\n"
            "  python download_weights.py --upload-to-hf <your-hf-username>/food-recipe-finder-weights\n"
            "or run `python -m src.run_pipeline` to train and quantize locally."
        )
        return False


def upload_weights(repo_id: str, weights_dir: Path = Path("weights")) -> None:
    api = HfApi()
    print(f"[weights] Creating/verifying Hugging Face model repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"[weights] Uploading {weights_dir} to https://huggingface.co/{repo_id} ...")
    api.upload_folder(
        folder_path=str(weights_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "runs/*", "*.prep.onnx"],
    )
    print(f"[weights] Upload complete! Link: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO, help="Hugging Face repo ID for model weights")
    parser.add_argument("--upload-to-hf", metavar="HF_REPO_ID", help="Upload ./weights to Hugging Face Hub")
    args = parser.parse_args()

    if args.upload_to_hf:
        upload_weights(args.upload_to_hf)
    else:
        ensure_weights(args.repo_id)
