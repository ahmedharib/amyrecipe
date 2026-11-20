"""Trained Neural Retrieval Baseline: Token Embeddings + MLP Projection (Area A / R3)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.retrieval.preprocess import normalize_text, parse_query, recipe_to_document
from src.retrieval.tfidf_retriever import apply_constraint_adjustment


class NeuralDualEncoder(nn.Module):
    """Feedforward embedding + MLP network trained with contrastive loss."""

    def __init__(self, vocab_size: int, emb_dim: int = 128, out_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean", padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, out_dim),
        )

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids, offsets)
        out = self.proj(emb)
        return F.normalize(out, p=2, dim=-1)


def _tokenize_for_nn(text: str) -> List[str]:
    return normalize_text(text).split()


def build_vocab(recipes: List[Dict[str, Any]], min_freq: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in recipes:
        for tok in _tokenize_for_nn(recipe_to_document(rec)):
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in sorted(counts.items()):
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def _encode_batch_inputs(texts: List[str], vocab: Dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    flat: List[int] = []
    offsets: List[int] = []
    unk_idx = vocab["<unk>"]
    for t in texts:
        offsets.append(len(flat))
        toks = _tokenize_for_nn(t)
        if not toks:
            flat.append(0)
        else:
            flat.extend(vocab.get(tok, unk_idx) for tok in toks)
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def _synthesize_training_pairs(recipes: List[Dict[str, Any]], seed: int = 42) -> List[tuple[str, int]]:
    """Generate diverse training (query, recipe_idx) pairs strictly from recipe metadata."""
    rng = random.Random(seed)
    pairs: List[tuple[str, int]] = []
    for idx, rec in enumerate(recipes):
        dish = rec["dish_category"].replace("_", " ")
        diets = [d.replace("_", " ") for d in rec.get("diet_tags", [])]
        ings = rec.get("ingredients", [])
        title = rec["title"]
        t_min = rec["prep_time_min"]

        pairs.append((title, idx))
        pairs.append((f"{dish} {title}", idx))
        if diets:
            diet_str = " ".join(rng.sample(diets, k=min(2, len(diets))))
            pairs.append((f"{diet_str} {dish}", idx))
            pairs.append((f"{diet_str} {dish} under {t_min + 5} minutes", idx))
        if ings:
            sample_ings = " ".join(rng.sample(ings, k=min(3, len(ings))))
            pairs.append((f"{dish} with {sample_ings}", idx))
        if t_min <= 25:
            pairs.append((f"quick {dish} under {t_min + 5} min", idx))
    return pairs


def train_neural_retriever(
    recipes: List[Dict[str, Any]],
    save_dir: str | Path = "weights/neural_retriever",
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 3e-3,
    seed: int = 42,
) -> Path:
    """Train the EmbeddingBag + MLP retriever on recipe-query pairs and save artifacts."""
    torch.manual_seed(seed)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab(recipes)
    docs = [recipe_to_document(r) for r in recipes]
    pairs = _synthesize_training_pairs(recipes, seed=seed)

    model = NeuralDualEncoder(vocab_size=len(vocab), emb_dim=128, out_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(pairs)
        total_loss = 0.0
        steps = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            if len(batch) < 2:
                continue
            q_texts = [p[0] for p in batch]
            d_texts = [docs[p[1]] for p in batch]

            q_ids, q_off = _encode_batch_inputs(q_texts, vocab)
            d_ids, d_off = _encode_batch_inputs(d_texts, vocab)

            q_emb = model(q_ids.to(device), q_off.to(device))
            d_emb = model(d_ids.to(device), d_off.to(device))

            logits = (q_emb @ d_emb.T) / 0.1
            labels = torch.arange(len(batch), device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        if epoch % 10 == 0 or epoch == epochs:
            print(f"[NeuralRetriever] Epoch {epoch}/{epochs} - loss={total_loss / max(1, steps):.4f}")

    # Precompute normalized recipe matrix on CPU for fast inference
    model_cpu = model.cpu().eval()
    with torch.no_grad():
        d_ids, d_off = _encode_batch_inputs(docs, vocab)
        doc_embs = model_cpu(d_ids, d_off).numpy()

    torch.save(model_cpu.state_dict(), save_path / "neural_retriever.pt")
    (save_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    np.save(save_path / "recipe_embeddings.npy", doc_embs)
    return save_path


class NeuralRetriever:
    """Inference wrapper for the trained EmbeddingBag + MLP retrieval model."""

    def __init__(self, recipes: List[Dict[str, Any]], model_dir: str | Path = "weights/neural_retriever") -> None:
        self.recipes = recipes
        model_path = Path(model_dir)
        self.vocab: Dict[str, int] = json.loads((model_path / "vocab.json").read_text(encoding="utf-8"))
        self.model = NeuralDualEncoder(vocab_size=len(self.vocab), emb_dim=128, out_dim=128)
        state = torch.load(model_path / "neural_retriever.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.doc_embs = np.load(model_path / "recipe_embeddings.npy")

    def encode_query(self, text: str) -> np.ndarray:
        q_ids, q_off = _encode_batch_inputs([text], self.vocab)
        with torch.no_grad():
            emb = self.model(q_ids, q_off).numpy()[0]
        return emb

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
