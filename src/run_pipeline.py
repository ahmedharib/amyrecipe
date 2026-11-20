"""End-to-end training, quantization, and benchmarking pipeline runner (Steps 2-6)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.eval.benchmark import main as run_benchmark
from src.retrieval.bert_retriever import train_bert_retriever
from src.retrieval.build_recipes import build_recipes_and_queries
from src.retrieval.neural_retriever import train_neural_retriever
from src.retrieval.quantize_bert import export_and_quantize_bert
from src.vision.prepare_data import prepare_food101_subset
from src.vision.quantize_cnn import export_and_quantize_cnn


def main() -> None:
    print("=== Step 2a: Prepare 10-class Food-101 subset ===")
    prepare_food101_subset(cleanup_raw=True)

    print("=== Step 2b: Build recipes.json & test_queries.json ===")
    build_recipes_and_queries()
    recipes = json.loads(Path("data/recipes.json").read_text(encoding="utf-8"))

    print("=== Step 3: Train Neural Retrieval Baseline ===")
    train_neural_retriever(recipes)

    print("=== Step 4: Train ScratchCNN & Fine-Tuned ResNet18 ===")
    subprocess.run([sys.executable, "-m", "src.vision.train_cnn"], check=True)

    print("=== Step 5: Fine-Tune BERT Sentence Embedder ===")
    train_bert_retriever(recipes)

    print("=== Step 6a: Quantize ConvNet & BERT to ONNX INT8 ===")
    export_and_quantize_cnn()
    export_and_quantize_bert()

    print("=== Step 6b: Run Unified Held-Out Benchmark ===")
    run_benchmark()


if __name__ == "__main__":
    main()
