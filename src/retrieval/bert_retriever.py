"""Transformer Sentence Embedding Retriever (BERT-based, Area C)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.retrieval.preprocess import parse_query, recipe_to_document
from src.retrieval.tfidf_retriever import apply_constraint_adjustment

DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build_st_training_examples(recipes: List[Dict[str, Any]], seed: int = 42) -> List[InputExample]:
    """Generate diverse natural-language training queries from recipe metadata."""
    rng = random.Random(seed)
    examples: List[InputExample] = []
    docs = [recipe_to_document(r) for r in recipes]

    for idx, rec in enumerate(recipes):
        dish = rec["dish_category"].replace("_", " ")
        diets = [d.replace("_", " ") for d in rec.get("diet_tags", [])]
        ings = rec.get("ingredients", [])
        t_min = rec["prep_time_min"]
        doc = docs[idx]

        examples.append(InputExample(texts=[rec["title"], doc]))
        if diets:
            d_str = " ".join(rng.sample(diets, k=min(2, len(diets))))
            examples.append(InputExample(texts=[f"{d_str} {dish} recipe", doc]))
            examples.append(InputExample(texts=[f"{d_str} {dish} under {t_min + 5} minutes", doc]))
        if t_min <= 25:
            examples.append(InputExample(texts=[f"quick and easy {dish} in {t_min + 5} min", doc]))
        if len(ings) >= 2:
            s_ings = ", ".join(rng.sample(ings, k=2))
            examples.append(InputExample(texts=[f"{dish} made with {s_ings}", doc]))
    return examples


def train_bert_retriever(
    recipes: List[Dict[str, Any]],
    base_model: str = DEFAULT_BASE_MODEL,
    save_dir: str | Path = "weights/bert_retriever",
    epochs: int = 2,
    batch_size: int = 32,
    seed: int = 42,
) -> Path:
    """Fine-tune a BERT-based SentenceTransformer with MultipleNegativesRankingLoss."""
    torch.manual_seed(seed)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(base_model, device=device)
    examples = _build_st_training_examples(recipes, seed=seed)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = max(10, int(len(loader) * epochs * 0.1))
    print(f"[BERTRetriever] Fine-tuning {base_model} on {len(examples)} pairs ({device})...")
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    model.save(str(save_path))

    # Precompute normalized recipe embeddings
    docs = [recipe_to_document(r) for r in recipes]
    doc_embs = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
    np.save(save_path / "recipe_embeddings.npy", doc_embs.astype(np.float32))
    print(f"[BERTRetriever] Saved fine-tuned model and recipe index to {save_path}")
    return save_path


class BertRetriever:
    """FP32 SentenceTransformer retrieval wrapper."""

    def __init__(self, recipes: List[Dict[str, Any]], model_dir: str | Path = "weights/bert_retriever") -> None:
        self.recipes = recipes
        model_path = Path(model_dir)
        self.model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
        self.doc_embs = np.load(model_path / "recipe_embeddings.npy")

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]

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
