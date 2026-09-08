"""Loss weighting -- the intervention under test.

Identity terms co-occur with abuse in the training distribution, so the model
learns the token as a shortcut. Upweighting the cell that *breaks* that
correlation -- non-toxic comments which do mention an identity -- should make the
shortcut more expensive than learning the semantics.

Three schemes so the ablation can separate them: ``none`` (baseline), ``class``
(inverse class frequency, the control that answers "would ordinary imbalance
handling have done this?") and ``subgroup`` (ours).

Identity annotations are used here and nowhere else at training time, and never
at inference. The model never sees a demographic feature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

SCHEMES = ("none", "class", "subgroup")


def compute_sample_weights(
    frame: pd.DataFrame,
    scheme: str,
    config: Any,
    alpha: float | None = None,
) -> np.ndarray:
    """Per-sample loss weights, computed from the training split alone."""
    if scheme not in SCHEMES:
        raise ValueError(
            f"unknown weighting scheme '{scheme}' (expected one of {', '.join(SCHEMES)})"
        )

    labels = frame["label"].to_numpy()
    weights = np.ones(len(frame), dtype=np.float32)

    if scheme == "none":
        _log_cells(frame, weights, scheme)
        return weights

    if scheme == "class":
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            logger.warning("only one class present in train; falling back to uniform weights")
            return weights
        weights[labels == 1] = n_neg / n_pos
        weights /= weights.mean()
        _log_cells(frame, weights, scheme)
        return weights

    resolved_alpha = float(config.get("mitigation.alpha", 4.0) if alpha is None else alpha)
    cfg_weights = config.get("mitigation.weights", {}) or {}

    def _weight(key: str, default: float) -> float:
        # 0.0 means "drop this cell", which is a legitimate ablation. Falling back
        # on truthiness would turn it into 1.0 and run a different experiment.
        value = cfg_weights.get(key)
        return default if value is None else float(value)

    w_background = _weight("background", 1.0)
    w_identity_toxic = _weight("identity_toxic", 1.0)
    # A null identity_nontoxic means "use alpha" -- alpha is the swept knob.
    configured_nontoxic = cfg_weights.get("identity_nontoxic")
    w_identity_nontoxic = (
        resolved_alpha if configured_nontoxic is None else float(configured_nontoxic)
    )

    has_identity = weighting_mask(frame, config)
    weights[:] = w_background
    weights[has_identity & (labels == 0)] = w_identity_nontoxic
    weights[has_identity & (labels == 1)] = w_identity_toxic

    logger.info(
        "subgroup weighting | alpha=%.2f | background=%.2f identity_nontoxic=%.2f "
        "identity_toxic=%.2f",
        resolved_alpha, w_background, w_identity_nontoxic, w_identity_toxic,
    )
    _log_cells(frame, weights, scheme, has_identity)
    return weights


def weighting_columns(config: Any) -> list[str] | None:
    """Identity columns the weighting is allowed to see, or None for all of them."""
    configured = config.get("mitigation.weighting_columns")
    return list(configured) if configured else None


def weighting_mask(frame: pd.DataFrame, config: Any) -> np.ndarray:
    """Which rows count as identity-bearing FOR THE WEIGHTING.

    Normally every annotated identity column, which is what ``any_identity``
    already holds. ``mitigation.weighting_columns`` narrows it to a subset, and
    that is the whole of the held-out-group experiment: the loss is told about
    some axes only, while the audit continues to measure every subgroup. If
    parity improves on the axes the weighting never saw, the intervention taught
    the model something about identity-bearing language in general rather than
    about the particular tokens it was paid to notice.

    The audit is deliberately NOT narrowed. Doing that would measure the arm on
    the groups it was tuned for, which is the one comparison that cannot fail.
    """
    columns = weighting_columns(config)
    if columns is None:
        return frame["any_identity"].to_numpy() == 1

    present = [c for c in columns if c in frame.columns]
    missing = sorted(set(columns) - set(present))
    if missing:
        logger.warning(
            "mitigation.weighting_columns names %d column(s) absent from the corpus, "
            "dropped: %s", len(missing), ", ".join(missing),
        )
    if not present:
        raise ValueError(
            "mitigation.weighting_columns matched no column in the corpus, so the "
            "weighting would see no identity at all and the arm would silently be "
            f"the baseline. Configured: {columns}"
        )
    mask = frame[present].to_numpy().sum(axis=1) > 0
    logger.info(
        "weighting sees %d of the identity columns (%s); the audit still measures all of them",
        len(present), ", ".join(present),
    )
    return mask


def is_uniform(weights: np.ndarray) -> bool:
    """True when every sample carries the same weight.

    Such a run is arithmetically the unweighted baseline whatever the scheme was
    called, so the ablation reuses the baseline instead of retraining it.
    """
    weights = np.asarray(weights, dtype=float)
    return bool(weights.size > 0 and np.allclose(weights, weights[0]))


def _log_cells(
    frame: pd.DataFrame,
    weights: np.ndarray,
    scheme: str,
    identity_mask: np.ndarray | None = None,
) -> None:
    """Record what each (identity x label) cell actually received.

    ``identity_mask`` is the weighting's own view of which rows are identity-bearing, so a
    narrowed run logs the cells it actually weighted rather than the ones the
    audit will report on.
    """
    labels = frame["label"].to_numpy()
    has_identity = (
        (frame["any_identity"].to_numpy() == 1) if identity_mask is None else identity_mask
    )
    cells = {
        "background_nontoxic": ~has_identity & (labels == 0),
        "background_toxic": ~has_identity & (labels == 1),
        "identity_nontoxic": has_identity & (labels == 0),
        "identity_toxic": has_identity & (labels == 1),
    }
    parts = [
        f"{name}: n={int(mask.sum()):,} w={weights[mask].mean():.2f}"
        for name, mask in cells.items()
        if mask.any()
    ]
    logger.info("weights [%s] | %s", scheme, " | ".join(parts))


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits under per-sample weights.

    Normalised by the **sum of weights**, not the batch size: dividing by batch
    size would make the gradient magnitude scale with alpha, so alpha would act
    partly as a learning rate and the ablation would measure that confound.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
        groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `groups` is accepted and unused so that the two objectives share one
        # call signature and the training loop needs no branch. A branch there
        # would be a second place for the arms to differ.
        del groups
        logits = logits.squeeze(-1)
        per_sample = self.elementwise(logits, targets.float())
        if weights is None:
            return per_sample.mean()
        weights = weights.to(per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp(min=self.eps)


# The four cells the static scheme weights, in the order `_log_cells` prints
# them. Group DRO learns a weight for each of these instead of being handed one,
# which is what makes the two objectives directly comparable.
GROUP_NAMES: tuple[str, ...] = (
    "background_nontoxic", "background_toxic", "identity_nontoxic", "identity_toxic",
)
OBJECTIVES = ("bce", "group_dro")


def compute_group_ids(frame: pd.DataFrame) -> np.ndarray:
    """Cell index per row: 2 * any_identity + label, matching ``GROUP_NAMES``."""
    labels = frame["label"].to_numpy().astype(np.int64)
    has_identity = (frame["any_identity"].to_numpy() == 1).astype(np.int64)
    return 2 * has_identity + labels


class GroupDROLoss(nn.Module):
    """Distributionally robust BCE over the (identity x label) cells.

    Sagawa et al. (2020). The static scheme is handed a weight vector chosen in
    advance; this one *learns* one online, raising the weight of whichever cell
    currently has the highest loss:

        q_g  <-  q_g * exp(eta * L_g),  renormalised,   loss = sum_g q_g * L_g

    That makes it the dynamic twin of our own method rather than a different
    method that happens to also be about fairness -- same four cells, same place
    in the pipeline, and the only difference is whether the weights were chosen
    by us or by the optimiser. The learned q is recorded at the end of training,
    so the paper can print what the algorithm decided next to what we assumed.

    `weights` are honoured as within-cell sample weights, so stacking the two
    interventions stays meaningful and nothing is silently dropped. With the
    uniform weights this arm actually runs under, that reduces to a plain mean.
    """

    def __init__(self, n_groups: int = 4, step_size: float = 0.01, eps: float = 1e-8) -> None:
        super().__init__()
        if n_groups < 2:
            raise ValueError(f"group DRO needs at least two groups, got {n_groups}")
        self.n_groups = int(n_groups)
        self.step_size = float(step_size)
        self.eps = eps
        self.elementwise = nn.BCEWithLogitsLoss(reduction="none")
        self.register_buffer("q", torch.full((self.n_groups,), 1.0 / self.n_groups))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
        groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if groups is None:
            raise ValueError(
                "GroupDROLoss needs a group id per sample. The dataset was built without "
                "one, so this run would silently be plain BCE under another name."
            )
        logits = logits.squeeze(-1)
        # float32 for the aggregation: under AMP the per-sample losses arrive as
        # float16, and index_add_ over a whole batch in half precision loses
        # enough to move the q update.
        per_sample = self.elementwise(logits, targets.float()).float()
        groups = groups.long()
        sample_weights = (
            torch.ones_like(per_sample) if weights is None else weights.float()
        )

        totals = torch.zeros(self.n_groups, device=per_sample.device, dtype=per_sample.dtype)
        counts = torch.zeros_like(totals)
        totals.index_add_(0, groups, per_sample * sample_weights)
        counts.index_add_(0, groups, sample_weights)

        # A cell missing from this batch keeps its q and contributes nothing.
        # Treating it as zero loss instead would drive its weight down for being
        # rare, which is the opposite of what the method is for.
        present = counts > 0
        cell_loss = torch.zeros_like(totals)
        cell_loss[present] = totals[present] / counts[present].clamp(min=self.eps)

        with torch.no_grad():
            q = self.q.to(per_sample.device).clone()
            q[present] = q[present] * torch.exp(self.step_size * cell_loss[present])
            q = q / q.sum().clamp(min=self.eps)
            self.q.copy_(q.to(self.q.device))

        return (q[present] * cell_loss[present]).sum()

    def learned_weights(self) -> dict[str, float]:
        """The final q, named, so it can be printed beside the static vector."""
        values = self.q.detach().cpu().tolist()
        return {name: float(v) for name, v in zip(GROUP_NAMES, values, strict=False)}


def build_criterion(config: Any, objective: str | None = None) -> nn.Module:
    """The training objective for ONE model.

    ``objective`` is passed in rather than read from the config, because both
    arms read the same config: reading it here would give the baseline the
    mitigated arm's objective and destroy the controlled pair. It falls back to
    the config only when a caller does not care which arm it is building.
    """
    objective = str(
        config.get("mitigation.objective", "bce") if objective is None else objective
    ).lower()
    if objective not in OBJECTIVES:
        raise ValueError(
            f"unknown mitigation.objective '{objective}' "
            f"(expected one of {', '.join(OBJECTIVES)})"
        )
    if objective == "group_dro":
        step_size = float(config.get("mitigation.dro.step_size", 0.01))
        logger.info("objective: group DRO over %d cells | step_size=%.4f",
                    len(GROUP_NAMES), step_size)
        return GroupDROLoss(n_groups=len(GROUP_NAMES), step_size=step_size)
    return WeightedBCEWithLogitsLoss()
