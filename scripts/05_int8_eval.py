"""Evaluate INT8 ONNX vs fp32 PyTorch embeddings on a subset of test pairs (full CPU corpus embedding is too slow)."""
import json, os, random, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval_utils import evaluate_at_threshold, load_corpus, load_pairs, pair_similarities

ROOT = Path(__file__).resolve().parent.parent
N_FUNCS = int(sys.argv[1]) if len(sys.argv) > 1 else 600

ids, funcs = load_corpus()
id_to_idx = {f: i for i, f in enumerate(ids)}
ft = np.load(ROOT / "models" / "finetuned_embeddings.npy")
thr = json.load(open(ROOT / "models" / "finetuned_threshold.json"))["threshold"]

random.seed(0)
pairs = load_pairs("test")
pos = [p for p in pairs if p[2]]
neg = [p for p in pairs if not p[2]]
random.shuffle(pos); random.shuffle(neg)
chosen = set()
sub = []
for p in pos[:3000] + neg[:3000]:
    new = {p[0], p[1]} - chosen
    if len(chosen) + len(new) > N_FUNCS:
        continue
    chosen |= {p[0], p[1]}
    sub.append(p)
chosen = sorted(chosen)
print(f"{len(chosen)} functions, {len(sub)} pairs ({sum(p[2] for p in sub)} positive)")

tok = AutoTokenizer.from_pretrained(ROOT / "models" / "onnx")
so = ort.SessionOptions()
so.intra_op_num_threads = os.cpu_count()
sess = ort.InferenceSession(str(ROOT / "models" / "onnx" / "model_int8.onnx"), so, providers=["CPUExecutionProvider"])

texts = [funcs[id_to_idx[i]] for i in chosen]
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
emb = np.zeros((len(texts), 768), dtype=np.float32)
t0 = time.time()
for s in range(0, len(order), 16):
    idx = order[s:s + 16]
    enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256, return_tensors="np")
    emb[idx] = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
dt = time.time() - t0
print(f"int8 CPU: {dt:.0f}s for {len(texts)} functions = {dt/len(texts)*1000:.0f} ms/function (batched)")

local = {f: i for i, f in enumerate(chosen)}
sims8, labels = pair_similarities(sub, local, emb)
sims32, _ = pair_similarities(sub, id_to_idx, ft)
res = {"subset": {"functions": len(chosen), "pairs": len(sub)},
       "int8": evaluate_at_threshold(sims8, labels, thr),
       "fp32": evaluate_at_threshold(sims32, labels, thr),
       "int8_ms_per_function": dt / len(texts) * 1000}
print(json.dumps(res, indent=2))
json.dump(res, open(ROOT / "results" / "int8_vs_fp32.json", "w"), indent=2)
