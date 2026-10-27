"""Duplicate-detection engine used by the Claude Code hook: tree-sitter -> INT8 ONNX embeddings -> FAISS."""
import json
import os
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.java_parser import extract_methods_from_file

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "models" / "onnx"
MIN_LINES = 4
THRESHOLD = float(os.environ.get("CLONE_THRESHOLD", "0.82"))
SKIP_DIRS = {".git", "node_modules", "target", "build", "out", ".idea", ".clone_index"}

_sess = _tok = None


def _load():
    global _sess, _tok
    if _sess is None:
        _tok = AutoTokenizer.from_pretrained(ONNX_DIR)
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count()
        _sess = ort.InferenceSession(str(ONNX_DIR / "model_int8.onnx"), so, providers=["CPUExecutionProvider"])
    return _sess, _tok


def embed(texts: list[str], bs: int = 16) -> np.ndarray:
    sess, tok = _load()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), 768), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256, return_tensors="np")
        out[idx] = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                   "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
    return out


def file_methods(path: str) -> list[dict]:
    res = []
    for m in extract_methods_from_file(path):
        if m.end_line - m.start_line + 1 >= MIN_LINES:
            res.append({"file": str(Path(path).resolve()), "name": m.name, "cls": m.class_name,
                        "start": m.start_line, "end": m.end_line, "code": m.code})
    return res


def build_index(project_dir: str) -> int:
    methods = []
    for dp, dns, fns in os.walk(project_dir):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn.endswith(".java"):
                methods += file_methods(os.path.join(dp, fn))
    idx_dir = Path(project_dir) / ".clone_index"
    idx_dir.mkdir(exist_ok=True)
    if methods:
        emb = embed([m["code"] for m in methods])
        index = faiss.IndexFlatIP(768)
        index.add(emb)
        faiss.write_index(index, str(idx_dir / "faiss.index"))
    json.dump([{k: v for k, v in m.items() if k != "code"} for m in methods], open(idx_dir / "meta.json", "w"))
    return len(methods)


def find_duplicates(project_dir: str, changed_file: str, k: int = 3) -> list[dict]:
    idx_dir = Path(project_dir) / ".clone_index"
    if not (idx_dir / "faiss.index").exists():
        return []
    index = faiss.read_index(str(idx_dir / "faiss.index"))
    meta = json.load(open(idx_dir / "meta.json"))
    new = file_methods(changed_file)
    if not new:
        return []
    this = str(Path(changed_file).resolve())
    D, I = index.search(embed([m["code"] for m in new]), min(k + 20, index.ntotal))
    hits = []
    for m, drow, irow in zip(new, D, I):
        for d, j in zip(drow, irow):
            o = meta[j]
            if o["file"] == this or d < THRESHOLD:
                continue
            hits.append({"method": f"{m['cls']}.{m['name']}", "lines": f"{m['start']}-{m['end']}",
                         "similar_to": f"{o['cls']}.{o['name']} ({o['file']}:{o['start']}-{o['end']})", "sim": float(d)})
            break
    return hits
