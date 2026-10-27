"""Shared helpers for threshold tuning + retrieval-style clone detection eval."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_corpus(path: Path | None = None) -> tuple[list[int], list[str]]:
    path = path or DATA_DIR / "corpus.jsonl"
    ids, funcs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["id"])
            funcs.append(row["func"])
    return ids, funcs


def load_pairs(split: str) -> list[tuple[int, int, bool]]:
    name = "valid" if split == "validation" else split
    path = DATA_DIR / f"{name}_pairs.jsonl"
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pairs.append((row["id1"], row["id2"], bool(row["label"])))
    return pairs


def pair_similarities(pairs: list[tuple[int, int, bool]], id_to_idx: dict, embeddings: np.ndarray):
    idx1 = np.array([id_to_idx[p[0]] for p in pairs])
    idx2 = np.array([id_to_idx[p[1]] for p in pairs])
    labels = np.array([p[2] for p in pairs], dtype=bool)
    sims = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)  # cosine, embeddings are L2-normalized
    return sims, labels


def best_threshold(sims: np.ndarray, labels: np.ndarray, n_steps: int = 200) -> tuple[float, dict]:
    lo, hi = float(sims.min()), float(sims.max())
    best_f1, best_t, best_stats = -1.0, 0.0, {}
    for t in np.linspace(lo, hi, n_steps):
        preds = sims >= t
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
            best_stats = {"precision": p, "recall": r, "f1": f1, "threshold": float(t)}
    return best_t, best_stats


def evaluate_at_threshold(sims: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = sims >= threshold
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1, "threshold": threshold, "n": len(labels), "n_pos": int(labels.sum())}
