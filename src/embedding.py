"""GraphCodeBERT sentence-embedding encoder (mean-pooled, L2-normalized)."""
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/graphcodebert-base"
MAX_LENGTH = 256


class CodeEncoder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None, max_length: int = MAX_LENGTH):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def encode(self, texts: list[str], batch_size: int = 32, normalize: bool = True, show_progress: bool = False) -> torch.Tensor:
        """Encode a list of code strings into (N, hidden) embeddings."""
        all_embs = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for i in iterator:
                batch = texts[i : i + batch_size]
                emb = self._encode_batch(batch)
                all_embs.append(emb.cpu())
        if was_training:
            self.model.train()
        result = torch.cat(all_embs, dim=0)
        if normalize:
            result = torch.nn.functional.normalize(result, p=2, dim=1)
        return result

    def encode_batch_grad(self, texts: list[str], normalize: bool = True) -> torch.Tensor:
        """Encode with gradients enabled, for training."""
        emb = self._encode_batch(texts)
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def _encode_batch(self, batch: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        hidden = out.last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
