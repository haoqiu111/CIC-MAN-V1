"""Controlled measurement perturbations for target-free robustness evaluation."""

from __future__ import annotations


def apply_perturbation(x, perturbation: str, *, seed: int = 0):
    """Apply a deterministic batch perturbation to a torch tensor.

    The input is expected to have shape ``[batch, channels, length]``. These
    perturbations simulate measurement mechanism changes at evaluation time;
    they never use labels or target-domain adaptation.
    """

    import torch

    if perturbation in {"clean", "none"}:
        return x

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    if perturbation.startswith("gaussian_snr_"):
        snr_db = float(perturbation.removeprefix("gaussian_snr_").replace("m", "-"))
        signal_power = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return x + noise * noise_power.sqrt()

    if perturbation.startswith("scale_"):
        factor = float(perturbation.removeprefix("scale_").replace("p", "."))
        return x * factor

    if perturbation.startswith("dropout_"):
        rate = float(perturbation.removeprefix("dropout_").replace("p", "."))
        keep = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) >= rate
        return x * keep

    if perturbation.startswith("impulse_"):
        rate = float(perturbation.removeprefix("impulse_").replace("p", "."))
        mask = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) < rate
        signs = torch.where(
            torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) < 0.5,
            -torch.ones_like(x),
            torch.ones_like(x),
        )
        amp = x.std(dim=-1, keepdim=True).clamp_min(1e-6) * 6.0
        return x + mask.to(x.dtype) * signs * amp

    if perturbation.startswith("harmonic_"):
        amp_factor = float(perturbation.removeprefix("harmonic_").replace("p", "."))
        length = x.shape[-1]
        t = torch.linspace(0.0, 1.0, steps=length, device=x.device, dtype=x.dtype)
        wave = torch.sin(2.0 * torch.pi * 8.0 * t).view(1, 1, length)
        amp = x.std(dim=-1, keepdim=True).clamp_min(1e-6) * amp_factor
        return x + amp * wave

    if perturbation.startswith("trend_"):
        amp_factor = float(perturbation.removeprefix("trend_").replace("p", "."))
        length = x.shape[-1]
        trend = torch.linspace(-1.0, 1.0, steps=length, device=x.device, dtype=x.dtype).view(1, 1, length)
        amp = x.std(dim=-1, keepdim=True).clamp_min(1e-6) * amp_factor
        return x + amp * trend

    raise ValueError(f"Unknown perturbation: {perturbation}")
