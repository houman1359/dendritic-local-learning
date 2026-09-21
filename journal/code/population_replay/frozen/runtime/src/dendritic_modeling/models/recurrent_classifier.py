"""
Classifier wrapper for recurrent models.

Returns raw logits (no LogSoftmax) to avoid double-softmax with nn.CrossEntropyLoss.
Implements predict() for compatibility with PerformanceAnalyzer.
"""

import torch

from dendritic_modeling.models.base import BaseModel


class RecurrentClassifier(BaseModel):
    """Classifier for recurrent models. Returns raw logits (no LogSoftmax).

    predict() uses argmax(dim=-1) which naturally handles both:
    - [B, C] -> [B]  (many-to-one, output_mode="last"/"mean")
    - [B, T, C] -> [B, T]  (many-to-many, output_mode="all")
    """

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns class indices. [B] for many-to-one, [B, T] for many-to-many."""
        with torch.no_grad():
            return self.forward(x, **kwargs).argmax(dim=-1)

    def eval_logits(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns [B, C] logits for analysis code. Reduces 3D via last timestep."""
        with torch.no_grad():
            out = self.forward(x, **kwargs)
            if out.dim() == 3:
                out = out[:, -1, :]
            return out
