"""Shared evaluation metrics and CPU latency benchmarking utilities (R3, R4)."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


def top_k_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    """Compute top-k classification accuracy from logits [N, C] and targets [N]."""
    top_preds = np.argsort(-logits, axis=1)[:, :k]
    correct = np.any(top_preds == targets[:, None], axis=1)
    return float(np.mean(correct))


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    """Fraction of relevant IDs retrieved in the top-k results (capped at 1.0 if binary hit)."""
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    hits = len(set(ranked_ids[:k]) & rel)
    return hits / len(rel)


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    """1.0 if any relevant ID appears in top-k, else 0.0."""
    rel = set(relevant_ids)
    return 1.0 if any(rid in rel for rid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    rel = set(relevant_ids)
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in rel:
            return 1.0 / rank
    return 0.0


def file_size_mb(path: str | Path) -> float:
    """Return file size in megabytes (MiB-based decimal MB: bytes / 1e6)."""
    p = Path(path)
    if not p.exists():
        return 0.0
    total = p.stat().st_size
    ext = Path(str(p) + ".data")
    if ext.exists():
        total += ext.stat().st_size
    return round(total / (1024 * 1024), 2)


def measure_latency_ms(fn: Callable[[int], None], n_runs: int = 120, warmup: int = 15) -> float:
    """Measure mean latency in milliseconds over n_runs after warmup."""
    for i in range(warmup):
        fn(i)
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn(i)
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(float(np.mean(times)), 2)


def get_hardware_summary() -> dict:
    """Return CPU/platform summary for reproducible benchmark reporting."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
