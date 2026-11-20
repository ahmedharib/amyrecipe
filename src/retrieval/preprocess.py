"""Classical NLP preprocessing, regex constraint extraction, and stemming (Area B)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "with", "in", "on", "of", "to",
    "me", "find", "show", "give", "want", "looking", "recipe", "recipes",
    "dish", "make", "cook", "something", "that", "is", "are", "i", "please",
}

_TIME_PATTERNS = [
    re.compile(r"(?:under|less\s+than|within|<=|below|max(?:imum)?)\s*(\d+)\s*(?:min(?:ute)?s?|m\b)", re.I),
    re.compile(r"(\d+)\s*(?:min(?:ute)?s?|m\b)\s*(?:or\s+less|max)", re.I),
    re.compile(r"\bin\s*(\d+)\s*(?:min(?:ute)?s?)\b", re.I),
]

_DIET_PATTERNS = {
    "vegetarian": re.compile(r"\b(?:vegetarian|veggie|meatless)\b", re.I),
    "vegan": re.compile(r"\b(?:vegan|plant[- ]based)\b", re.I),
    "gluten_free": re.compile(r"\b(?:gluten[- ]?free|celiac|no\s+gluten)\b", re.I),
    "dairy_free": re.compile(r"\b(?:dairy[- ]?free|lactose[- ]?free|no\s+dairy)\b", re.I),
    "spicy": re.compile(r"\b(?:spicy|hot|chili|fiery)\b", re.I),
    "high_protein": re.compile(r"\b(?:high[- ]?protein|protein[- ]rich)\b", re.I),
}

_QUICK_PATTERN = re.compile(r"\b(?:quick|fast|express|weeknight|easy)\b", re.I)


@dataclass
class ParsedQuery:
    raw_query: str
    normalized_text: str
    stemmed_text: str
    tokens: List[str] = field(default_factory=list)
    max_time_min: Optional[int] = None
    diet_constraints: List[str] = field(default_factory=list)
    dish_hint: Optional[str] = None


def normalize_text(text: str) -> str:
    """Lowercase, expand common abbreviations, and strip non-alphanumeric noise."""
    text = text.lower().strip()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"\bmins?\b", "minutes", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stem_tokens(text: str, drop_stopwords: bool = True) -> List[str]:
    """Tokenize normalized text and apply Porter stemming."""
    norm = normalize_text(text)
    tokens = norm.split()
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return [_STEMMER.stem(t) for t in tokens]


def extract_constraints(query: str) -> tuple[Optional[int], List[str]]:
    """Extract max preparation time (minutes) and dietary/style tags via regex."""
    max_time: Optional[int] = None
    for pat in _TIME_PATTERNS:
        m = pat.search(query)
        if m:
            max_time = int(m.group(1))
            break

    if max_time is None and _QUICK_PATTERN.search(query):
        max_time = 25

    diets: List[str] = []
    for tag, pat in _DIET_PATTERNS.items():
        if pat.search(query):
            diets.append(tag)

    # Vegan implies vegetarian for filtering
    if "vegan" in diets and "vegetarian" not in diets:
        diets.append("vegetarian")

    return max_time, sorted(diets)


def parse_query(query: str, dish_hint: Optional[str] = None) -> ParsedQuery:
    """Full classical NLP query preprocessing pipeline."""
    max_time, diets = extract_constraints(query)
    combined = f"{dish_hint.replace('_', ' ')} {query}" if dish_hint else query
    norm = normalize_text(combined)
    tokens = stem_tokens(combined, drop_stopwords=True)
    return ParsedQuery(
        raw_query=query,
        normalized_text=norm,
        stemmed_text=" ".join(tokens),
        tokens=tokens,
        max_time_min=max_time,
        diet_constraints=diets,
        dish_hint=dish_hint,
    )


def recipe_to_document(recipe: dict) -> str:
    """Flatten a recipe record into searchable text."""
    parts = [
        recipe["title"],
        recipe["dish_category"].replace("_", " "),
        " ".join(recipe.get("diet_tags", [])).replace("_", " "),
        " ".join(recipe.get("ingredients", [])),
        recipe.get("description", ""),
        f"{recipe.get('prep_time_min', '')} minutes",
    ]
    return " ".join(parts)
