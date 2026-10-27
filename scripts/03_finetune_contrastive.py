"""Fine-tune GraphCodeBERT with a contrastive loss on BigCloneBench train pairs
so that FAISS cosine-similarity retrieval actually separates clones from
non-clones (the pretrained baseline collapses everything to ~0.97+ similarity).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.embedding import CodeEncoder
from src.eval_utils import best_threshold, evaluate_at_threshold, load_corpus, load_pairs, pair_similarities

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FT_DIR = MODELS_DIR / "finetuned"
FT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_SUBSET = 30_000       # subsample from 901k train pairs (kept 50/50 balanced) - sized for ~1-2hr on a 4GB laptop GPU
BATCH_SIZE = 6              # pairs per step -> 12 sequences/forward (4GB VRAM headroom)
GRAD_ACCUM = 5              # effective batch 30 pairs
EPOCHS = 2
LR = 2e-5
MARGIN = 0.4
MAX_LENGTH = 256

random.seed(SEED)
torch.manual_seed(SEED)


class PairDataset(Dataset):
    def __init__(self, pairs, id_to_code):
        self.pairs = pairs
        self.id_to_code = id_to_code

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        id1, id2, label = self.pairs[idx]
        return self.id_to_code[id1], self.id_to_code[id2], float(label)


def collate(batch):
    c1, c2, labels = zip(*batch)
    return list(c1), list(c2), torch.tensor(labels, dtype=torch.float32)


def contrastive_loss(cos_sim: torch.Tensor, labels: torch.Tensor, margin: float = MARGIN) -> torch.Tensor:
    pos_loss = labels * (1 - cos_sim).pow(2)
    neg_loss = (1 - labels) * torch.clamp(cos_sim - margin, min=0).pow(2)
    return (pos_loss + neg_loss).mean()


def run_validation(encoder: CodeEncoder, ids, funcs, id_to_idx):
    embeddings = encoder.encode(funcs, batch_size=32, normalize=True).numpy().astype("float32")
    valid_pairs = load_pairs("validation")
    sims, labels = pair_similarities(valid_pairs, id_to_idx, embeddings)
    threshold, stats = best_threshold(sims, labels)
    return embeddings, threshold, stats


def main():
    ids, funcs = load_corpus()
    id_to_idx = {fid: i for i, fid in enumerate(ids)}
    id_to_code = dict(zip(ids, funcs))

    train_pairs_all = load_pairs("train")
    random.shuffle(train_pairs_all)
    train_pairs = train_pairs_all[:TRAIN_SUBSET]
    print(f"Training on {len(train_pairs)} / {len(train_pairs_all)} train pairs")

    dataset = PairDataset(train_pairs, id_to_code)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    encoder = CodeEncoder(max_length=MAX_LENGTH)
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(encoder.device == "cuda"))

    print("Pre-finetune validation (sanity check vs baseline):")
    _, pre_threshold, pre_stats = run_validation(encoder, ids, funcs, id_to_idx)
    print(json.dumps(pre_stats, indent=2))

    best_f1 = pre_stats["f1"]
    best_threshold_value = pre_threshold
    history = []

    step = 0
    for epoch in range(EPOCHS):
        encoder.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (c1, c2, labels) in enumerate(loader):
            labels = labels.to(encoder.device)
            with torch.amp.autocast("cuda", enabled=(encoder.device == "cuda")):
                emb1 = encoder.encode_batch_grad(c1, normalize=True)
                emb2 = encoder.encode_batch_grad(c2, normalize=True)
                cos_sim = (emb1 * emb2).sum(dim=1)
                loss = contrastive_loss(cos_sim, labels) / GRAD_ACCUM

            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM
            step += 1

            if step % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"epoch {epoch} step {i+1}/{len(loader)} loss={running_loss/(i+1):.4f} elapsed={elapsed:.0f}s")

        print(f"Epoch {epoch} done in {time.time()-t0:.0f}s, avg loss={running_loss/len(loader):.4f}")

        embeddings, threshold, stats = run_validation(encoder, ids, funcs, id_to_idx)
        print(f"Epoch {epoch} validation:", json.dumps(stats, indent=2))
        history.append({"epoch": epoch, **stats})

        if stats["f1"] > best_f1:
            best_f1 = stats["f1"]
            best_threshold_value = threshold
            encoder.model.save_pretrained(FT_DIR)
            encoder.tokenizer.save_pretrained(FT_DIR)
            np.save(MODELS_DIR / "finetuned_embeddings.npy", embeddings)
            with open(MODELS_DIR / "finetuned_ids.json", "w") as f:
                json.dump(ids, f)
            with open(MODELS_DIR / "finetuned_threshold.json", "w") as f:
                json.dump({"threshold": best_threshold_value}, f)
            print(f"  -> new best (F1={best_f1:.4f}), checkpoint saved to {FT_DIR}")

    with open(RESULTS_DIR / "finetune_history.json", "w") as f:
        json.dump({"pre_finetune": pre_stats, "epochs": history, "best_f1": best_f1, "best_threshold": best_threshold_value}, f, indent=2)

    print(f"\nBest validation F1: {best_f1:.4f} at threshold {best_threshold_value:.4f}")


if __name__ == "__main__":
    main()
