"""Classical TF-IDF + cosine similarity + regex constraint retrieval baseline (Area B)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.preprocess import parse_query, recipe_to_document, stem_tokens


def apply_constraint_adjustment(
    base_scores: np.ndarray,
    recipes: List[Dict[str, Any]],
    max_time_min: Optional[int],
    diet_constraints: List[str],
    dish_hint: Optional[str] = None,
) -> np.ndarray:
    """Filter to predicted dish (if given) and penalize time/diet constraint violations."""
    adjusted = base_scores.astype(np.float64).copy()
    for idx, rec in enumerate(recipes):
        if dish_hint and rec["dish_category"] != dish_hint:
            adjusted[idx] = -1e9
            continue

        if max_time_min is not None and rec["prep_time_min"] > max_time_min:
            adjusted[idx] -= 0.40

        if diet_constraints:
            rec_diets = set(rec.get("diet_tags", []))
            missing = sum(1 for d in diet_constraints if d not in rec_diets)
            if missing > 0:
                adjusted[idx] -= 0.45 * missing
    return adjusted


class TfidfRetriever:
    """TF-IDF retrieval over Porter-stemmed recipe documents."""

    def __init__(self, recipes: List[Dict[str, Any]]) -> None:
        self.recipes = recipes
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda s: stem_tokens(s, drop_stopwords=True),
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        docs = [recipe_to_document(r) for r in recipes]
        self.doc_matrix = self.vectorizer.fit_transform(docs)

    def search(
        self,
        query: str,
        dish_hint: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        parsed = parse_query(query, dish_hint=dish_hint)
        q_text = f"{dish_hint.replace('_', ' ')} {query}" if dish_hint else query
        q_vec = self.vectorizer.transform([q_text])
        raw_sims = cosine_similarity(q_vec, self.doc_matrix).ravel()
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

        return {
            "parsed_query": {
                "normalized_text": parsed.normalized_text,
                "stemmed_text": parsed.stemmed_text,
                "max_time_min": parsed.max_time_min,
                "diet_constraints": parsed.diet_constraints,
                "dish_hint": parsed.dish_hint,
            },
            "results": results,
        }
