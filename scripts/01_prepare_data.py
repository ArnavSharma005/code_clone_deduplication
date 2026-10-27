"""Load BigCloneBench (Java) from CodeXGLUE and build:
  - data/corpus.jsonl       : {id, func} unique Java functions
  - data/train_pairs.jsonl  : {id1, id2, label}
  - data/valid_pairs.jsonl  : {id1, id2, label}
  - data/test_pairs.jsonl   : {id1, id2, label}
"""
import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def main():
    ds = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")

    corpus = {}
    pair_counts = {}
    for split in ("train", "validation", "test"):
        rows = ds[split]
        # Columnar access (whole-column slicing backed by Arrow) is orders of
        # magnitude faster than `for row in rows`, which pays per-row Python
        # dict-construction overhead across all 901k/415k/415k examples.
        id1_col = rows["id1"]
        id2_col = rows["id2"]
        label_col = rows["label"]
        func1_col = rows["func1"]
        func2_col = rows["func2"]

        pairs_path = DATA_DIR / f"{'valid' if split == 'validation' else split}_pairs.jsonl"
        n_pos = 0
        with pairs_path.open("w", encoding="utf-8") as pf:
            lines = []
            for id1, id2, label, f1, f2 in zip(id1_col, id2_col, label_col, func1_col, func2_col):
                label = bool(label)
                if id1 not in corpus:
                    corpus[id1] = f1
                if id2 not in corpus:
                    corpus[id2] = f2
                lines.append(json.dumps({"id1": id1, "id2": id2, "label": label}))
                n_pos += int(label)
            pf.write("\n".join(lines) + "\n")
        pair_counts[split] = (len(id1_col), n_pos)

    corpus_path = DATA_DIR / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as cf:
        for fid, code in corpus.items():
            cf.write(json.dumps({"id": fid, "func": code}) + "\n")

    print(f"Unique functions: {len(corpus)}")
    for split, (total, pos) in pair_counts.items():
        print(f"{split}: {total} pairs, {pos} positive ({pos / total:.4%}), {total - pos} negative")


if __name__ == "__main__":
    main()
