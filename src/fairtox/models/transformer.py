"""The classifier: a pretrained encoder with a single-logit head.

Not ``AutoModelForSequenceClassification``, because it hides the loss and every
intervention in this study lives in the loss.

Position 0 of the last hidden state, not a pooler: DistilBERT ships no pooler and
some checkpoints initialise theirs randomly, while position 0 is defined
identically for every backbone in scope -- which is why swapping in
distilroberta-base needed no code change at all.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ToxicityClassifier(nn.Module):
    """Pretrained encoder -> dropout -> Linear(hidden, 1). Returns raw logits."""

    def __init__(
        self,
        backbone: str = "distilbert-base-uncased",
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone_name = backbone
        self.encoder = AutoModel.from_pretrained(backbone)

        config = self.encoder.config
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
        if hidden_size is None:
            raise AttributeError(f"cannot determine hidden size for backbone '{backbone}'")

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(hidden_size), 1)

        if freeze_encoder:
            # A memory/time escape hatch for a cramped allocation. Not used for
            # the headline result: freezing the encoder changes what is being
            # measured, so it would have to be reported as a different model.
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            logger.warning("encoder is FROZEN -- only the classification head will train")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_token))


def build_model(config: Any) -> ToxicityClassifier:
    """Construct the classifier described by ``config``."""
    model = ToxicityClassifier(
        backbone=str(config.get("model.backbone", "distilbert-base-uncased")),
        dropout=float(config.get("model.dropout", 0.2)),
        freeze_encoder=bool(config.get("model.freeze_encoder", False)),
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "model %s | %s parameters (%s trainable)",
        model.backbone_name, f"{total:,}", f"{trainable:,}",
    )
    return model
