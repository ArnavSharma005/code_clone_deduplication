"""Baseline: pretrained GraphCodeBERT embeddings + FAISS retrieval, no fine-tuning.

Encodes every unique function once, builds a FAISS cosine-similarity index
(for the retrieval-at-scale story), then evaluates pairwise clone detection
by looking up each labeled pair's similarity, tuning a threshold on the
validation split and reporting P/R/F1 on test.
"""
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.embedding import CodeEncoder
from src.eval_utils import best_threshold, evaluate_at_threshold, load_corpus, load_pairs, pair_similarities

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    ids, funcs = load_corpus()
    print(f"Corpus: {len(ids)} unique functions")

    encoder = CodeEncoder()
    print(f"Encoding on {encoder.device} ...")
    t0 = time.time()
    embeddings = encoder.encode(funcs, batch_size=32, normalize=True, show_progress=True).numpy().astype("float32")
    print(f"Encoded {len(ids)} functions in {time.time() - t0:.1f}s, shape={embeddings.shape}")

    np.save(MODELS_DIR / "baseline_embeddings.npy", embeddings)
    with open(MODELS_DIR / "baseline_ids.json", "w") as f:
        json.dump(ids, f)

    # FAISS cosine-similarity index (vectors are L2-normalized -> inner product = cosine)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(MODELS_DIR / "baseline_faiss.index"))

    # Sanity-check the retrieval path: query a random function, inspect nearest neighbours.
    D, I = index.search(embeddings[:3], k=5)
    print("Sample FAISS self-search (should rank itself #1 with sim~1.0):")
    for row_d, row_i in zip(D, I):
        print("  sims:", np.round(row_d, 3).tolist(), "ids:", [ids[j] for j in row_i])

    id_to_idx = {fid: i for i, fid in enumerate(ids)}

    valid_pairs = load_pairs("validation")
    test_pairs = load_pairs("test")
    print(f"Validation pairs: {len(valid_pairs)}, Test pairs: {len(test_pairs)}")

    valid_sims, valid_labels = pair_similarities(valid_pairs, id_to_idx, embeddings)
    test_sims, test_labels = pair_similarities(test_pairs, id_to_idx, embeddings)

    threshold, valid_stats = best_threshold(valid_sims, valid_labels)
    test_stats = evaluate_at_threshold(test_sims, test_labels, threshold)

    print("\n=== Baseline (pretrained GraphCodeBERT, no fine-tuning) ===")
    print("Validation (threshold search):", json.dumps(valid_stats, indent=2))
    print("Test (threshold from validation):", json.dumps(test_stats, indent=2))

    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump({"validation": valid_stats, "test": test_stats}, f, indent=2)


if __name__ == "__main__":
    main()
