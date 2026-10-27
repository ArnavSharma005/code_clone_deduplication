"""Exportable wrapper: RoBERTa/GraphCodeBERT encoder + mean-pool + L2-normalize
as a single torch.nn.Module, so the ONNX graph is self-contained (no Python
post-processing needed at inference time in the hook)."""
import torch
import torch.nn as nn
from transformers import AutoModel


class PooledEncoder(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
