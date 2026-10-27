"""Export fine-tuned encoder to ONNX, INT8-quantize, verify, benchmark, and eval."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval_utils import evaluate_at_threshold, load_corpus, load_pairs, pair_similarities
from src.onnx_encoder import PooledEncoder

ROOT = Path(__file__).resolve().parent.parent
FT_DIR = ROOT / "models" / "finetuned"
ONNX_DIR = ROOT / "models" / "onnx"
ONNX_DIR.mkdir(exist_ok=True)
FP32 = ONNX_DIR / "model.onnx"
INT8 = ONNX_DIR / "model_int8.onnx"
MAX_LEN = 256


def embed_onnx(sess, tok, texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
        out.append(sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                   "attention_mask": enc["attention_mask"].astype(np.int64)})[0])
    return np.concatenate(out)


def main():
    tok = AutoTokenizer.from_pretrained(FT_DIR)
    tok.save_pretrained(ONNX_DIR)
    model = PooledEncoder(str(FT_DIR)).eval()
    enc = tok(["int a = b + c;", "public void foo() { return; }"], padding=True, return_tensors="pt")
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"]), str(FP32),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "embedding": {0: "b"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)
    print(f"fp32 {FP32.stat().st_size/1e6:.0f}MB  int8 {INT8.stat().st_size/1e6:.0f}MB")

    so = ort.SessionOptions()
    s32 = ort.InferenceSession(str(FP32), so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(INT8), so, providers=["CPUExecutionProvider"])

    ids, funcs = load_corpus()
    with torch.no_grad():
        ref = model(enc["input_ids"], enc["attention_mask"]).numpy()
    got = s32.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    print("max |onnx - torch| =", float(np.abs(ref - got).max()))

    sample = funcs[:64]
    for name, s in (("fp32", s32), ("int8", s8)):
        embed_onnx(s, tok, sample[:8])
        t0 = time.time()
        embed_onnx(s, tok, sample, bs=1)
        print(f"{name} CPU single-function latency: {(time.time()-t0)/len(sample)*1000:.0f} ms")

    thr = json.load(open(ROOT / "models" / "finetuned_threshold.json"))["threshold"]
    id_to_idx = {f: i for i, f in enumerate(ids)}
    results = {}
    for name, s in (("int8", s8),):
        t0 = time.time()
        emb = embed_onnx(s, tok, funcs)
        print(f"{name} embedded corpus in {time.time()-t0:.0f}s")
        np.save(ONNX_DIR / f"embeddings_{name}.npy", emb)
        for split in ("validation", "test"):
            sims, labels = pair_similarities(load_pairs(split), id_to_idx, emb)
            results[f"{name}_{split}"] = evaluate_at_threshold(sims, labels, thr)
            print(name, split, results[f"{name}_{split}"])
    ft = np.load(ROOT / "models" / "finetuned_embeddings.npy")
    sims, labels = pair_similarities(load_pairs("test"), id_to_idx, ft)
    results["fp32_test"] = evaluate_at_threshold(sims, labels, thr)
    print("fp32 (pytorch) test", results["fp32_test"])
    json.dump(results, open(ROOT / "results" / "final_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
