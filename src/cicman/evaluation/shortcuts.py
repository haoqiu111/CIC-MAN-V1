"""Label-conditioned shortcut perturbations for reversal experiments."""

from __future__ import annotations


def apply_label_shortcut(
    x,
    labels,
    *,
    mode: str = "none",
    num_classes: int | None = None,
    amplitude: float = 1.0,
    base_frequency: float = 5.0,
    frequency_step: float = 3.0,
):
    """Inject a label-conditioned harmonic shortcut.

    ``correlated`` assigns each class its own harmonic frequency. ``reversed``
    assigns the opposite class frequency, creating a controlled shortcut
    reversal at evaluation time. ``neutral`` adds the same frequency to all
    samples, breaking the shortcut-label relation without removing the
    measurement artifact.
    """

    import torch

    if mode in {"none", "clean", ""}:
        return x
    if num_classes is None:
        num_classes = int(labels.detach().max().item()) + 1 if labels.numel() else 1
    if num_classes <= 1:
        return x

    class_ids = labels.detach().long()
    if mode == "correlated":
        shortcut_ids = class_ids
    elif mode == "reversed":
        shortcut_ids = (num_classes - 1) - class_ids
    elif mode == "neutral":
        shortcut_ids = torch.zeros_like(class_ids)
    else:
        raise ValueError(f"Unknown shortcut mode: {mode}")

    length = x.size(-1)
    t = torch.linspace(0.0, 1.0, steps=length, device=x.device, dtype=x.dtype).view(1, 1, length)
    frequencies = base_frequency + frequency_step * shortcut_ids.to(dtype=x.dtype).view(-1, 1, 1)
    waves = torch.sin(2.0 * torch.pi * frequencies * t)
    amp = x.std(dim=-1, keepdim=True).clamp_min(1e-6) * amplitude
    return x + amp * waves
