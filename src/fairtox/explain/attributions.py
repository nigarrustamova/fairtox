"""Token attribution via Layer Integrated Gradients.

One question: in a false positive on benign identity-bearing text, how much of
the toxicity logit sits on the identity token, and does mitigation move it?

High attribution on "muslim" means that token moved *this* prediction on *this*
example. It is not counterfactual proof and not a statement about the model's
reasoning in general, so this is reported as supporting analysis only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class AttributionExplainer:
    """Layer Integrated Gradients over the token-embedding layer."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: str = "cpu") -> None:
        try:
            from captum.attr import LayerIntegratedGradients
        except ImportError as exc:  # optional-tier dependency
            raise ImportError(
                "the attributions stage needs captum, which is not installed.\n"
                "  pip install captum\n"
                "Or exclude it by running with --tier must."
            ) from exc

        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_layer = self._find_embeddings(model)
        self.explainer = LayerIntegratedGradients(self._forward, self.embedding_layer)

    @staticmethod
    def _find_embeddings(model: torch.nn.Module) -> torch.nn.Module:
        """Locate the word-embedding module across backbone families."""
        encoder = getattr(model, "encoder", model)
        embeddings = getattr(encoder, "embeddings", None)
        if embeddings is None:
            raise AttributeError(
                "could not locate an embeddings module on the backbone; "
                "Integrated Gradients needs the layer to attribute to"
            )
        return getattr(embeddings, "word_embeddings", embeddings)

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)

    def explain(self, text: str, max_length: int = 128, n_steps: int = 32) -> dict[str, Any]:
        """Per-token attribution toward the toxicity logit for one comment."""
        encoded = self.tokenizer(
            text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # The all-PAD sequence is the conventional neutral reference. CLS and SEP
        # are restored so the baseline is still a well-formed input to the
        # encoder rather than a sequence the model has never seen.
        pad_id = self.tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_id if pad_id is not None else 0)
        n_tokens = int(attention_mask[0].sum().item())
        baseline_ids[0, 0] = input_ids[0, 0]
        baseline_ids[0, max(n_tokens - 1, 0)] = input_ids[0, max(n_tokens - 1, 0)]

        attributions, delta = self.explainer.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        # Sum over embedding dimensions, then normalise so examples compare.
        scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        norm = float(np.linalg.norm(scores))
        if norm > 0:
            scores = scores / norm

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][:n_tokens])
        scores = scores[:n_tokens]

        with torch.no_grad():
            prob = torch.sigmoid(self._forward(input_ids, attention_mask)).item()

        return {
            "text": text,
            "prob": float(prob),
            "tokens": list(tokens),
            "attributions": [float(s) for s in scores],
            "convergence_delta": float(delta.item()),
        }


def identity_attribution_mass(
    explanation: dict[str, Any], identity_terms: set[str]
) -> dict[str, Any]:
    """Share of total attribution mass sitting on identity tokens.

    This is the number the comparison turns on: if mitigation works, the same
    sentence should route less of its toxicity signal through the identity term
    after training than before.
    """
    # "##ing" -> "ing" for wordpiece continuations, "ĠMuslim" -> "Muslim" for
    # byte-level BPE. removeprefix, not lstrip: lstrip("##") would also eat the
    # hash of a token like "#hashtag".
    #
    # The prefixes come off BEFORE the lowercasing, and that order is the whole
    # point. RoBERTa's word-boundary marker is U+0120 (capital G with dot above);
    # lowercasing it first yields U+0121, which removeprefix("Ġ") then never
    # matches. Measured on distilroberta-base: two identity tokens matched across
    # twenty explanations instead of eighty, so the reported identity share was an
    # artefact of the tokenizer rather than a property of the model.
    tokens = [str(t).removeprefix("##").removeprefix("Ġ").lower() for t in explanation["tokens"]]
    scores = np.abs(np.asarray(explanation["attributions"], dtype=float))
    total = float(scores.sum())
    if total == 0:
        return {"identity_mass": 0.0, "identity_share": 0.0, "matched_tokens": []}

    matched = [i for i, token in enumerate(tokens) if token in identity_terms]
    identity_mass = float(scores[matched].sum()) if matched else 0.0
    return {
        "identity_mass": identity_mass,
        "identity_share": float(identity_mass / total),
        "matched_tokens": [explanation["tokens"][i] for i in matched],
    }
