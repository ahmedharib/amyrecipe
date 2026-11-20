"""Export fine-tuned BERT sentence embedder to ONNX and apply dynamic INT8 quantization (Area D)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer

from src.retrieval.preprocess import parse_query
from src.retrieval.tfidf_retriever import apply_constraint_adjustment


class _EncoderExportWrapper(nn.Module):
    """Wrap HuggingFace Transformer + mean pooling + L2 normalization for clean ONNX export."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return torch.nn.functional.normalize(pooled, p=2, dim=-1)


def export_and_quantize_bert(model_dir: Path = Path("weights/bert_retriever")) -> tuple[Path, Path]:
    fp32_onnx = model_dir / "bert_fp32.onnx"
    int8_onnx = model_dir / "bert_int8.onnx"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    hf_model = AutoModel.from_pretrained(str(model_dir)).cpu().eval()
    wrapper = _EncoderExportWrapper(hf_model).eval()

    sample = tokenizer("quick vegetarian pizza under 30 minutes", return_tensors="pt", padding=True, truncation=True)
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_onnx),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "sentence_embedding": {0: "batch_size"},
        },
        opset_version=18,
    )
    print(f"Exported FP32 BERT ONNX to {fp32_onnx}")

    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized INT8 BERT ONNX saved to {int8_onnx}")
    return fp32_onnx, int8_onnx


class QuantizedBertRetriever:
    """CPU ONNX Runtime INT8 retrieval engine."""

    def __init__(
        self,
        recipes: List[Dict[str, Any]],
        model_dir: str | Path = "weights/bert_retriever",
        onnx_filename: str = "bert_int8.onnx",
    ) -> None:
        self.recipes = recipes
        model_path = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path / onnx_filename),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.doc_embs = np.load(model_path / "recipe_embeddings.npy")

    def encode_query(self, text: str) -> np.ndarray:
        enc = self.tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
        outputs = self.session.run(
            ["sentence_embedding"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )
        return outputs[0][0]

    def search(self, query: str, dish_hint: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        parsed = parse_query(query, dish_hint=dish_hint)
        q_text = f"{dish_hint.replace('_', ' ')} {query}" if dish_hint else query
        q_emb = self.encode_query(q_text)
        raw_sims = self.doc_embs @ q_emb
        final_scores = apply_constraint_adjustment(
            raw_sims,
            self.recipes,
            max_time_min=parsed.max_time_min,
            diet_constraints=parsed.diet_constraints,
            dish_hint=dish_hint,
        )
        top_idx = np.argsort(-final_scores)[:top_k]
        results = []
        for idx in top_idx:
            rec = dict(self.recipes[int(idx)])
            rec["score"] = float(round(float(final_scores[int(idx)]), 4))
            rec["raw_similarity"] = float(round(float(raw_sims[int(idx)]), 4))
            results.append(rec)
        return {"parsed_query": parsed.__dict__, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("weights/bert_retriever"))
    args = parser.parse_args()
    export_and_quantize_bert(args.model_dir)
