"""Unified benchmark over the held-out Food-101 test set and recipe test queries (R3, R4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets

from src.eval.metrics import (
    file_size_mb,
    get_hardware_summary,
    hit_at_k,
    measure_latency_ms,
    recall_at_k,
    reciprocal_rank,
    top_k_accuracy,
)
from src.retrieval.bert_retriever import BertRetriever
from src.retrieval.neural_retriever import NeuralRetriever
from src.retrieval.quantize_bert import QuantizedBertRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.vision.dataset import get_transforms
from src.vision.models import ScratchCNN, build_transfer_resnet18


def benchmark_vision(
    data_dir: Path = Path("data/food101_10class"),
    vision_dir: Path = Path("weights/vision"),
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    _, eval_tf = get_transforms(224)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    classes = test_ds.classes

    # Preload sample tensors for single-image CPU latency benchmarking (>= 100 runs)
    sample_inputs = [test_ds[i][0].unsqueeze(0).numpy().astype(np.float32) for i in range(0, len(test_ds), max(1, len(test_ds) // 40))]
    sample_tensors = [torch.from_numpy(x) for x in sample_inputs]

    all_targets = np.array([y for _, y in test_ds.samples], dtype=np.int64)
    rows: List[Dict[str, Any]] = []
    failure_examples: List[Dict[str, Any]] = []

    # 1. ScratchCNN (FP32, CPU)
    scratch = ScratchCNN(num_classes=len(classes))
    scratch.load_state_dict(torch.load(vision_dir / "scratch_cnn.pt", map_location="cpu", weights_only=True))
    scratch.eval()
    scratch_logits = []
    with torch.no_grad():
        for x, _ in loader:
            scratch_logits.append(scratch(x).numpy())
    scratch_logits_np = np.concatenate(scratch_logits, axis=0)
    scratch_lat = measure_latency_ms(
        lambda i: scratch(sample_tensors[i % len(sample_tensors)]), n_runs=120, warmup=15
    )
    rows.append({
        "pipeline": "Vision",
        "model": "ScratchCNN (4-block)",
        "precision": "FP32",
        "top1_acc": round(top_k_accuracy(scratch_logits_np, all_targets, k=1), 4),
        "top5_acc": round(top_k_accuracy(scratch_logits_np, all_targets, k=5), 4),
        "size_mb": file_size_mb(vision_dir / "scratch_cnn.pt"),
        "latency_ms": scratch_lat,
    })

    # 2. Fine-tuned ResNet18 (FP32, CPU)
    resnet = build_transfer_resnet18(num_classes=len(classes), pretrained=False)
    resnet.load_state_dict(torch.load(vision_dir / "resnet18_finetuned.pt", map_location="cpu", weights_only=True))
    resnet.eval()
    resnet_logits = []
    with torch.no_grad():
        for x, _ in loader:
            resnet_logits.append(resnet(x).numpy())
    resnet_logits_np = np.concatenate(resnet_logits, axis=0)
    resnet_lat = measure_latency_ms(
        lambda i: resnet(sample_tensors[i % len(sample_tensors)]), n_runs=120, warmup=15
    )
    rows.append({
        "pipeline": "Vision",
        "model": "ResNet18 Fine-Tuned",
        "precision": "FP32",
        "top1_acc": round(top_k_accuracy(resnet_logits_np, all_targets, k=1), 4),
        "top5_acc": round(top_k_accuracy(resnet_logits_np, all_targets, k=5), 4),
        "size_mb": file_size_mb(vision_dir / "resnet18_finetuned.pt"),
        "latency_ms": resnet_lat,
    })

    # 3. Quantized ResNet18 (ONNX INT8, CPU)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(
        str(vision_dir / "resnet18_int8.onnx"),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    ort_logits = []
    for x, _ in loader:
        out = ort_sess.run(["logits"], {"input": x.numpy().astype(np.float32)})[0]
        ort_logits.append(out)
    ort_logits_np = np.concatenate(ort_logits, axis=0)
    ort_lat = measure_latency_ms(
        lambda i: ort_sess.run(["logits"], {"input": sample_inputs[i % len(sample_inputs)]}),
        n_runs=120,
        warmup=15,
    )
    rows.append({
        "pipeline": "Vision",
        "model": "ResNet18 Quantized (ONNX)",
        "precision": "INT8",
        "top1_acc": round(top_k_accuracy(ort_logits_np, all_targets, k=1), 4),
        "top5_acc": round(top_k_accuracy(ort_logits_np, all_targets, k=5), 4),
        "size_mb": file_size_mb(vision_dir / "resnet18_int8.onnx"),
        "latency_ms": ort_lat,
    })

    # Collect up to 3 honest vision failure examples from the quantized model
    preds = np.argmax(ort_logits_np, axis=1)
    for idx in range(len(all_targets)):
        if preds[idx] != all_targets[idx]:
            img_path, true_idx = test_ds.samples[idx]
            failure_examples.append({
                "type": "vision",
                "image_path": img_path,
                "true_label": classes[true_idx],
                "predicted_label": classes[int(preds[idx])],
            })
            if len(failure_examples) >= 3:
                break

    return rows, failure_examples


def benchmark_retrieval(
    recipes_path: Path = Path("data/recipes.json"),
    queries_path: Path = Path("data/test_queries.json"),
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    tfidf = TfidfRetriever(recipes)
    neural = NeuralRetriever(recipes, "weights/neural_retriever")
    bert = BertRetriever(recipes, "weights/bert_retriever")
    qbert = QuantizedBertRetriever(recipes, "weights/bert_retriever", "bert_int8.onnx")

    def eval_retriever(name: str, precision: str, retriever: Any, size_mb: float) -> tuple[Dict[str, Any], List[Dict]]:
        recalls, hits, mrrs = [], [], []
        misses = []
        for q in queries:
            res = retriever.search(q["query"], dish_hint=q.get("dish_hint"), top_k=5)
            ranked_ids = [r["id"] for r in res["results"]]
            rel_ids = q["relevant_ids"]
            r5 = recall_at_k(ranked_ids, rel_ids, k=5)
            h5 = hit_at_k(ranked_ids, rel_ids, k=5)
            rr = reciprocal_rank(ranked_ids, rel_ids)
            recalls.append(r5)
            hits.append(h5)
            mrrs.append(rr)
            if h5 == 0.0 and len(misses) < 3:
                misses.append({
                    "type": "retrieval",
                    "model": name,
                    "query": q["query"],
                    "dish_hint": q.get("dish_hint"),
                    "expected_ids": rel_ids[:3],
                    "returned_ids": ranked_ids[:3],
                })

        lat = measure_latency_ms(
            lambda i: retriever.search(queries[i % len(queries)]["query"], dish_hint=queries[i % len(queries)].get("dish_hint"), top_k=5),
            n_runs=120,
            warmup=15,
        )
        return {
            "pipeline": "Retrieval",
            "model": name,
            "precision": precision,
            "recall_at_5": round(float(np.mean(recalls)), 4),
            "hit_at_5": round(float(np.mean(hits)), 4),
            "mrr": round(float(np.mean(mrrs)), 4),
            "size_mb": size_mb,
            "latency_ms": lat,
        }, misses

    rows = []
    failure_examples = []

    tfidf_size = round(len(tfidf.vectorizer.vocabulary_) * 8 / (1024 * 1024), 3)
    neural_size = round(
        file_size_mb("weights/neural_retriever/neural_retriever.pt")
        + file_size_mb("weights/neural_retriever/recipe_embeddings.npy"),
        2,
    )
    bert_fp32_size = file_size_mb("weights/bert_retriever/bert_fp32.onnx")
    bert_int8_size = file_size_mb("weights/bert_retriever/bert_int8.onnx")

    for name, prec, ret, sz in [
        ("Regex + TF-IDF Cosine", "N/A", tfidf, tfidf_size),
        ("EmbeddingBag + MLP (Trained)", "FP32", neural, neural_size),
        ("Fine-Tuned MiniLM (BERT)", "FP32", bert, bert_fp32_size),
        ("Quantized MiniLM (ONNX)", "INT8", qbert, bert_int8_size),
    ]:
        row, misses = eval_retriever(name, prec, ret, sz)
        rows.append(row)
        failure_examples.extend(misses)

    return rows, failure_examples


def main() -> None:
    vision_rows, vision_failures = benchmark_vision()
    retrieval_rows, retrieval_failures = benchmark_retrieval()

    report = {
        "hardware": get_hardware_summary(),
        "vision_results": vision_rows,
        "retrieval_results": retrieval_rows,
        "failure_examples": vision_failures + retrieval_failures,
    }
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "benchmark_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
