"""Device selection and VRAM accounting.

The brief caps peak GPU memory per team, and exceeding the allocation is an
automatic deduction. Two numbers matter and they are not the same:

``allocated``  what tensors actually hold.
``reserved``   what the caching allocator has taken from the driver, which is
               what ``nvidia-smi`` shows and what another team on the same card
               cannot use.

Reporting only ``allocated`` under-states the footprint, so both are recorded.
"""

from __future__ import annotations

from typing import Any


def resolve_device(preference: str = "auto") -> str:
    """Return a concrete torch device string."""
    try:
        import torch
    except ImportError:
        return "cpu"

    preference = (preference or "auto").lower()
    if preference != "auto":
        if preference.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> dict[str, Any]:
    """Hardware facts for the paper's Experimental Setup section."""
    info: dict[str, Any] = {"device": device}
    try:
        import torch
    except ImportError:
        return info

    info["torch"] = torch.__version__
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_gb": round(properties.total_memory / 1024**3, 2),
                "cuda": torch.version.cuda,
            }
        )
    return info


def reset_peak_memory(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory(device: str) -> dict[str, float | None]:
    """Peak allocated and reserved VRAM in GB, or None on CPU."""
    if not device.startswith("cuda"):
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}
    import torch

    if not torch.cuda.is_available():
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }
