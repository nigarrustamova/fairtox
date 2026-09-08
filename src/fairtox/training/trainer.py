"""The fine-tuning loop.

Built around three constraints. The booked window can end mid-training, so every
epoch writes ``last.pt`` with model, optimizer, scheduler, scaler *and the run
history* -- the training curve in the paper is then the whole curve, not the part
after the restart. Peak VRAM is capped, so allocated and reserved peaks are both
recorded. Results must reproduce, so nothing is fitted outside the training split.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import classification_metrics, summarise
from ..utils import device as device_utils
from ..utils.logging_utils import get_logger
from .losses import build_criterion

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    epochs: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    best_metric: float = float("-inf")
    epochs_without_gain: int = 0
    stopped_early: bool = False
    resumed_from_epoch: int | None = None
    train_seconds: float = 0.0
    peak_allocated_gb: float | None = None
    peak_reserved_gb: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "stopped_early": self.stopped_early,
            "resumed_from_epoch": self.resumed_from_epoch,
            "train_seconds": round(self.train_seconds, 1),
            "peak_allocated_gb": self.peak_allocated_gb,
            "peak_reserved_gb": self.peak_reserved_gb,
        }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: str | None = None,
        objective: str | None = None,
    ) -> None:
        self.config = config
        self.device = device or device_utils.resolve_device(str(config.get("run.device", "auto")))
        self.device_type = self.device.split(":")[0]
        self.model = model.to(self.device)
        # `.to(device)` matters for group DRO, whose learned weight vector is a
        # buffer: the criterion is not part of `model`, so nothing else moves it.
        self.criterion = build_criterion(config, objective).to(self.device)

        self.epochs = int(config.get("training.num_epochs", 3))
        self.accum_steps = max(1, int(config.get("training.gradient_accumulation_steps", 1)))
        self.max_grad_norm = float(config.get("training.max_grad_norm", 1.0))
        self.log_every = int(config.get("training.log_every_n_steps", 100))
        self.patience = int(config.get("training.early_stopping_patience", 2))
        self.monitor = str(config.get("training.early_stopping_metric", "val_macro_f1"))

        # Which epoch's weights are kept. "best" is the study; "last" is the
        # control that takes checkpoint selection out of the comparison. The two
        # arms do not peak on the same epoch, and "the arms differ only in the
        # weighting" has to be true of the KEPT model, not only of the recipe.
        self.select = str(config.get("training.select_checkpoint", "best")).lower()
        if self.select not in ("best", "last"):
            raise ValueError(
                "training.select_checkpoint must be 'best' or 'last', "
                f"got '{self.select}'"
            )
        self.threshold = float(config.get("evaluation.classification_threshold", 0.5))
        self.memory_budget_gb = config.get("training.memory_budget_gb")

        # AMP is a CUDA-only win here; on CPU it costs more than it saves.
        self.amp_enabled = (
            bool(config.get("training.mixed_precision", True)) and self.device_type == "cuda"
        )
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.amp_enabled)

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

    # -- setup ---------------------------------------------------------------

    def _build_optimizer(self, num_training_steps: int) -> None:
        from transformers import get_linear_schedule_with_warmup

        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Biases and LayerNorm gains are conventionally exempt from decay.
            is_no_decay = any(key in name for key in ("bias", "LayerNorm.weight", "layer_norm"))
            (no_decay if is_no_decay else decay).append(param)

        weight_decay = float(self.config.get("training.weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(self.config.get("training.learning_rate", 2e-5)),
        )
        warmup = int(num_training_steps * float(self.config.get("training.warmup_ratio", 0.1)))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup, num_training_steps=num_training_steps
        )

    def _optimizer_steps_per_epoch(self, loader: DataLoader) -> int:
        """Optimizer steps one epoch will actually take.

        Ceiling, not floor: the loop also steps on the final partial accumulation
        group at the end of an epoch. Using the floor would under-count, the
        linear schedule would run past ``num_training_steps``, and the last few
        steps of the run would silently train at a learning rate of zero.
        """
        return max(1, math.ceil(len(loader) / self.accum_steps))

    # -- training ------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        resume: bool = True,
    ) -> TrainingHistory:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._build_optimizer(self._optimizer_steps_per_epoch(train_loader) * self.epochs)

        history = TrainingHistory()
        start_epoch = self._maybe_resume(checkpoint_dir, history) if resume else 0
        if start_epoch >= self.epochs:
            logger.info("checkpoint is already at epoch %d of %d; nothing to train",
                        start_epoch, self.epochs)
            self._load_weights(checkpoint_dir / "best.pt")
            return history

        device_utils.reset_peak_memory(self.device)
        info = device_utils.describe_device(self.device)
        gpu_note = (
            f" ({info['gpu_name']}, {info['gpu_total_memory_gb']} GB)" if "gpu_name" in info else ""
        )
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "training on %s%s | amp=%s | accum=%d | %s trainable params",
            self.device, gpu_note, self.amp_enabled, self.accum_steps, f"{trainable:,}",
        )

        started = time.perf_counter()
        for epoch in range(start_epoch, self.epochs):
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_probs, val_labels = self.predict(val_loader)
            val_metrics = classification_metrics(val_labels, val_probs, self.threshold)

            record = {"epoch": epoch, "train_loss": train_loss}
            record.update({"val_" + key: value for key, value in val_metrics.items()})
            history.epochs.append(record)
            logger.info(
                "epoch %d | train_loss=%.4f | %s", epoch, train_loss, summarise(val_metrics)
            )

            if val_metrics.get("degenerate"):
                # Not fatal: a collapsed epoch early in training is normal. It is
                # logged because a run that stays collapsed produces perfect
                # subgroup parity, and that must never reach the paper unnoticed.
                logger.warning(
                    "epoch %d predicted a SINGLE CLASS on validation. If this persists, "
                    "the model has collapsed and its parity numbers are an artefact.",
                    epoch,
                )

            current = record.get(self.monitor)
            if current is None:
                raise KeyError(
                    f"early-stopping metric '{self.monitor}' was not produced by evaluation. "
                    f"Available: {', '.join(sorted(k for k in record if k.startswith('val_')))}"
                )

            if current > history.best_metric:
                history.best_metric = float(current)
                history.epochs_without_gain = 0
                if self.select == "best":
                    history.best_epoch = epoch
                    self._save_checkpoint(
                        checkpoint_dir / "best.pt", epoch, history, resumable=False
                    )
                    logger.info("new best %s=%.4f -> best.pt", self.monitor, current)
            else:
                history.epochs_without_gain += 1

            if self.select == "last":
                # The fixed-epoch control. `best_metric` still records the true
                # peak for the log, but the kept weights are the final epoch's,
                # so both arms are compared after the same amount of training.
                history.best_epoch = epoch
                self._save_checkpoint(checkpoint_dir / "best.pt", epoch, history, resumable=False)

            history.train_seconds = time.perf_counter() - started
            self._save_checkpoint(checkpoint_dir / "last.pt", epoch, history, resumable=True)
            self._check_memory_budget()

            # Early stopping is a form of selection, so it is off in the control.
            if self.select == "best" and history.epochs_without_gain >= self.patience:
                history.stopped_early = True
                logger.info(
                    "early stopping: no gain in %s for %d epoch(s)", self.monitor, self.patience
                )
                break

        history.train_seconds = time.perf_counter() - started
        history.epochs.sort(key=lambda record: record["epoch"])
        peaks = device_utils.peak_memory(self.device)
        history.peak_allocated_gb = peaks["peak_allocated_gb"]
        history.peak_reserved_gb = peaks["peak_reserved_gb"]
        if history.peak_reserved_gb is not None:
            logger.info(
                "peak VRAM: %.2f GB allocated / %.2f GB reserved (reserved is what nvidia-smi "
                "shows and what other teams cannot use)",
                history.peak_allocated_gb, history.peak_reserved_gb,
            )

        self._load_weights(checkpoint_dir / "best.pt")

        # The run finished, so the resume state is dead weight (AdamW moments make
        # it roughly 3x the size of the weights). best.pt has all anything
        # downstream needs.
        resume_state = checkpoint_dir / "last.pt"
        if resume_state.exists():
            resume_state.unlink()
            logger.info("training complete; removed last.pt resume state")

        return history

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        if self.optimizer is None:
            raise RuntimeError("optimizer was not built before training")

        self.model.train()
        total_loss, seen = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            weights = batch["weight"].to(self.device, non_blocking=True)
            groups = batch["group"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device_type, enabled=self.amp_enabled):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels, weights, groups)

            self.scaler.scale(loss / self.accum_steps).backward()

            if (step + 1) % self.accum_steps == 0 or (step + 1) == num_batches:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                # AMP skips the optimizer step when a gradient is non-finite and
                # lowers the scale to retry. The schedule must not advance on a step
                # the weights never took, or the LR the paper reports is not the LR
                # the model was trained at.
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= scale_before:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            seen += batch_size

            if self.log_every and step % self.log_every == 0:
                logger.info(
                    "  epoch %d step %d/%d | loss=%.4f | lr=%.2e",
                    epoch, step, num_batches, loss.item(), self.scheduler.get_last_lr()[0],
                )

        # NOTE: this is the weight-normalised loss, so its scale depends on the
        # weighting scheme. It is a progress signal, not a quantity to compare
        # across arms -- the arms are compared on held-out metrics, never on this.
        return total_loss / max(seen, 1)

    def _check_memory_budget(self) -> None:
        """Warn as soon as the run approaches the allocation, not afterwards."""
        if not self.memory_budget_gb:
            return
        reserved = device_utils.peak_memory(self.device)["peak_reserved_gb"]
        if reserved is None:
            return
        budget = float(self.memory_budget_gb)
        if reserved > budget:
            logger.warning(
                "PEAK RESERVED VRAM %.2f GB EXCEEDS THE %.1f GB BUDGET. Stop, lower "
                "training.batch_size and raise training.gradient_accumulation_steps by the "
                "same factor (same effective batch, smaller footprint).",
                reserved, budget,
            )
        elif reserved > 0.85 * budget:
            logger.warning("peak reserved VRAM %.2f GB is within 15%% of the %.1f GB budget",
                           reserved, budget)

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return (probabilities, labels) for every row in ``loader``, in order.

        Order is preserved because evaluation loaders are built with
        ``shuffle=False`` and never drop a batch; the audit rejoins these
        probabilities to identity flags by position.
        """
        self.model.eval()
        probs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device_type, enabled=self.amp_enabled):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs.append(torch.sigmoid(logits.squeeze(-1).float()).cpu().numpy())
            labels.append(batch["label"].numpy())
        return np.concatenate(probs), np.concatenate(labels)

    # -- checkpointing -------------------------------------------------------

    def _save_checkpoint(
        self, path: Path, epoch: int, history: TrainingHistory, *, resumable: bool
    ) -> None:
        """Write a checkpoint.

        Full resume state is ~3x the weights (AdamW keeps two moments), so only
        ``last.pt`` pays that cost and it is deleted on clean completion. The
        history travels with it: without that, a resumed run would report only
        the epochs after the restart.
        """
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "best_metric": history.best_metric,
            "best_epoch": history.best_epoch,
            "config": self.config.as_dict(),
        }
        if resumable:
            payload.update(
                {
                    "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
                    "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                    "scaler_state": self.scaler.state_dict(),
                    "history_epochs": history.epochs,
                    "epochs_without_gain": history.epochs_without_gain,
                    "train_seconds": history.train_seconds,
                }
            )
        torch.save(payload, path)

    def _maybe_resume(self, checkpoint_dir: Path, history: TrainingHistory) -> int:
        path = checkpoint_dir / "last.pt"
        if not path.exists():
            return 0

        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state") and self.optimizer:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scheduler_state") and self.scheduler:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            self.scaler.load_state_dict(payload["scaler_state"])

        history.best_metric = float(payload.get("best_metric", float("-inf")))
        history.best_epoch = int(payload.get("best_epoch", -1))
        history.epochs = list(payload.get("history_epochs", []))
        history.epochs_without_gain = int(payload.get("epochs_without_gain", 0))
        history.train_seconds = float(payload.get("train_seconds", 0.0))

        resume_epoch = int(payload["epoch"]) + 1
        history.resumed_from_epoch = resume_epoch
        logger.info(
            "resuming from %s at epoch %d (best %s=%.4f, %d earlier epoch(s) recovered)",
            path.name, resume_epoch, self.monitor, history.best_metric, len(history.epochs),
        )
        return resume_epoch

    def _load_weights(self, path: Path) -> None:
        if not path.exists():
            logger.warning("no %s to load; keeping final-epoch weights", path.name)
            return
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        logger.info("loaded best weights from epoch %d", payload.get("epoch", -1))
