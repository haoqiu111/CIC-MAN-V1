#!/usr/bin/env python3
"""Diagnose a complete CIC-MAN intervention view bank with source-only probes.

The script keeps the main CIC-MAN model untouched. It builds interpretable
signal views and measures whether each view preserves fault-health semantics,
leaks source-domain identity, and exposes fault-mechanism fidelity cues.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal


def add_src_to_path() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[2]
    sys.path.insert(0, str(project_dir / "src"))


add_src_to_path()

from cicman.data.dataset import WindowIndexDataset  # noqa: E402
from cicman.evaluation.metrics import accuracy, macro_f1  # noqa: E402


VIEW_NAMES = [
    "raw_core",
    "envelope",
    "stft",
    "wavelet",
    "order",
    "denoise",
    "filterbank",
]

FEATURE_NAMES = [
    "rms",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "spectral_entropy",
    "spectral_concentration",
    "spectral_peak_ratio",
    "spectral_high_ratio",
    "envelope_peak_ratio",
    "envelope_harmonicity",
    "stft_concentration",
    "stft_ridge_stability",
    "wavelet_low_energy",
    "wavelet_mid_energy",
    "wavelet_high_energy",
    "wavelet_energy_entropy",
    "order_peak_ratio",
    "order_harmonicity",
    "denoise_residual_ratio",
    "fault_mechanism_fidelity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True, help="Source train window CSV.")
    parser.add_argument("--eval-index", type=Path, required=True, help="Source held-out validation window CSV.")
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-items", type=int, default=12000)
    parser.add_argument("--max-eval-items", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def scalar(value: object) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def sample_indices(rows: list[dict[str, str]], max_items: int | None, seed: int) -> list[int]:
    indices = list(range(len(rows)))
    if max_items is None or max_items <= 0 or len(indices) <= max_items:
        return indices
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(str(row.get("dataset_id", "")), str(row.get("label_id", "")))].append(idx)
    for values in groups.values():
        rng.shuffle(values)
    per_group = max(1, max_items // max(1, len(groups)))
    selected: list[int] = []
    leftovers: list[int] = []
    for values in groups.values():
        selected.extend(values[:per_group])
        leftovers.extend(values[per_group:])
    if len(selected) < max_items:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_items - len(selected)])
    if len(selected) > max_items:
        rng.shuffle(selected)
        selected = selected[:max_items]
    return sorted(selected)


def moving_average(x: np.ndarray, kernel: int) -> np.ndarray:
    kernel = min(kernel, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if kernel <= 1:
        return x.astype(np.float32)
    pad = kernel // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    weights = np.ones(kernel, dtype=np.float64) / float(kernel)
    return np.convolve(padded, weights, mode="valid")[: len(x)].astype(np.float32)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-6
    return ((x - med) / (1.4826 * mad)).astype(np.float32)


def haar_band_energies(x: np.ndarray) -> tuple[float, float, float, float]:
    current = x.astype(np.float64)
    energies = []
    for _ in range(3):
        if len(current) < 4:
            break
        even = current[0::2]
        odd = current[1::2]
        length = min(len(even), len(odd))
        approx = (even[:length] + odd[:length]) * 0.5
        detail = (even[:length] - odd[:length]) * 0.5
        energies.append(float(np.mean(detail**2)))
        current = approx
    low = float(np.mean(current**2)) if len(current) else 0.0
    while len(energies) < 3:
        energies.append(0.0)
    total = low + sum(energies) + 1e-12
    ratios = [value / total for value in energies]
    low_ratio = low / total
    probs = np.asarray([*ratios, low_ratio], dtype=np.float64)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / math.log(len(probs))
    high, mid, low_detail = ratios[0], ratios[1], ratios[2]
    return low_ratio, mid + low_detail, high, entropy


def parse_rpm(metadata: dict[str, str]) -> float | None:
    candidates = [
        metadata.get("rotation_speed_rpm", ""),
        metadata.get("condition_id", ""),
        metadata.get("speed_profile_id", ""),
        metadata.get("recording_id", ""),
    ]
    for value in candidates:
        text = str(value)
        match = re.search(r"(?<!\d)(\d{3,5})(?:\s*rpm)?(?!\d)", text, flags=re.IGNORECASE)
        if match:
            rpm = float(match.group(1))
            if 100.0 <= rpm <= 30000.0:
                return rpm
    return None


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = x.astype(np.float64) - float(np.mean(x))
    window = np.hanning(len(centered))
    mag = np.abs(np.fft.rfft(centered * window))
    freq = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    power = mag**2
    return freq, mag, power


def entropy_from_power(power: np.ndarray) -> float:
    total = float(power.sum())
    if total <= 1e-12 or len(power) <= 1:
        return 0.0
    prob = power / total
    return -float(np.sum(prob * np.log(prob + 1e-12))) / math.log(len(prob))


def harmonicity(power: np.ndarray, peak_index: int, max_harmonics: int = 4) -> float:
    if peak_index <= 0 or len(power) <= 2:
        return 0.0
    total = float(power.sum()) + 1e-12
    harmonic_power = 0.0
    for harmonic in range(1, max_harmonics + 1):
        idx = peak_index * harmonic
        if idx >= len(power):
            break
        left = max(0, idx - 1)
        right = min(len(power), idx + 2)
        harmonic_power += float(power[left:right].sum())
    return harmonic_power / total


def base_metrics(x: np.ndarray, fs: float) -> dict[str, float]:
    centered = x.astype(np.float64) - float(np.mean(x))
    abs_mean = float(np.mean(np.abs(centered))) + 1e-12
    rms = float(np.sqrt(np.mean(centered**2) + 1e-12))
    std = float(np.std(centered) + 1e-12)
    peak_abs = float(np.max(np.abs(centered))) if len(centered) else 0.0
    kurtosis = float(np.mean((centered / std) ** 4))
    freq, mag, power = spectrum(centered, fs)
    usable = power[1:] if len(power) > 1 else power
    peak_idx = int(np.argmax(usable) + 1) if len(power) > 1 else 0
    spectral_entropy = entropy_from_power(usable)
    high_start = max(1, int(len(power) * 0.35))
    high_ratio = float(power[high_start:].sum() / (power.sum() + 1e-12)) if len(power) else 0.0
    peak_ratio = float(power[peak_idx] / (np.mean(usable) + 1e-12)) if len(usable) and peak_idx < len(power) else 0.0
    return {
        "rms": rms,
        "kurtosis": kurtosis,
        "crest_factor": peak_abs / (rms + 1e-12),
        "impulse_factor": peak_abs / abs_mean,
        "shape_factor": rms / abs_mean,
        "spectral_entropy": spectral_entropy,
        "spectral_concentration": 1.0 - spectral_entropy,
        "spectral_peak_ratio": peak_ratio,
        "spectral_high_ratio": high_ratio,
        "_peak_index": float(peak_idx),
    }


def envelope_metrics(x: np.ndarray, fs: float) -> dict[str, float]:
    analytic = signal.hilbert(x.astype(np.float64))
    envelope = np.abs(analytic)
    _, _, power = spectrum(envelope - float(np.mean(envelope)), fs)
    usable = power[1:] if len(power) > 1 else power
    peak_idx = int(np.argmax(usable) + 1) if len(power) > 1 and len(usable) else 0
    peak_ratio = float(power[peak_idx] / (np.mean(usable) + 1e-12)) if len(usable) and peak_idx < len(power) else 0.0
    return {
        "envelope_peak_ratio": peak_ratio,
        "envelope_harmonicity": harmonicity(power, peak_idx),
    }


def stft_metrics(x: np.ndarray, fs: float) -> dict[str, float]:
    nperseg = min(512, max(64, len(x) // 4))
    _, _, zxx = signal.stft(x.astype(np.float64), fs=fs, nperseg=nperseg, noverlap=nperseg // 2, boundary=None)
    power = np.abs(zxx) ** 2
    if power.size == 0:
        return {"stft_concentration": 0.0, "stft_ridge_stability": 0.0}
    total = float(power.sum()) + 1e-12
    concentration = float(power.max() / (power.mean() + 1e-12))
    ridge = np.argmax(power, axis=0) if power.ndim == 2 and power.shape[1] else np.asarray([0])
    stability = 1.0 / (1.0 + float(np.std(ridge)))
    return {
        "stft_concentration": concentration / max(1.0, total / (power.mean() + 1e-12)),
        "stft_ridge_stability": stability,
    }


def view_signal(x: np.ndarray, view: str) -> np.ndarray:
    raw = x.astype(np.float32)
    smooth_short = moving_average(raw, 9)
    smooth_mid = moving_average(raw, 33)
    smooth_long = moving_average(raw, 129)
    highpass = raw - smooth_short
    if view == "raw_core":
        return raw
    if view == "envelope":
        return np.abs(signal.hilbert(highpass.astype(np.float64))).astype(np.float32)
    if view == "stft":
        _, _, zxx = signal.stft(raw.astype(np.float64), nperseg=256, noverlap=128, boundary=None)
        band_energy = np.sqrt((np.abs(zxx) ** 2).mean(axis=1) + 1e-12)
        return robust_z(np.interp(np.linspace(0, len(band_energy) - 1, num=len(raw)), np.arange(len(band_energy)), band_energy))
    if view == "wavelet":
        detail1 = raw[0::2] - raw[1::2]
        if len(detail1) < 2:
            return highpass.astype(np.float32)
        return robust_z(np.interp(np.linspace(0, len(detail1) - 1, num=len(raw)), np.arange(len(detail1)), detail1))
    if view == "order":
        return robust_z(raw - smooth_long)
    if view == "denoise":
        kernel = min(101, len(raw) - 1 if len(raw) % 2 == 0 else len(raw))
        kernel = max(5, kernel if kernel % 2 == 1 else kernel - 1)
        denoised = signal.savgol_filter(raw.astype(np.float64), window_length=kernel, polyorder=3, mode="interp")
        residual = raw.astype(np.float64) - denoised
        return robust_z(residual)
    if view == "filterbank":
        mid_band = smooth_short - smooth_mid
        low_band = smooth_mid - smooth_long
        return np.sqrt(highpass**2 + 0.5 * mid_band**2 + 0.25 * low_band**2 + 1e-6).astype(np.float32)
    raise ValueError(f"Unknown view: {view}")


def feature_row(x: np.ndarray, fs: float, rpm: float | None, view: str) -> dict[str, float]:
    v = view_signal(x, view)
    metrics = base_metrics(v, fs)
    metrics.update(envelope_metrics(v, fs))
    metrics.update(stft_metrics(v, fs))
    low, mid, high, wavelet_entropy = haar_band_energies(v)
    metrics["wavelet_low_energy"] = low
    metrics["wavelet_mid_energy"] = mid
    metrics["wavelet_high_energy"] = high
    metrics["wavelet_energy_entropy"] = wavelet_entropy
    _, _, power = spectrum(v, fs)
    peak_idx = int(metrics.pop("_peak_index", 0.0))
    if rpm is not None and rpm > 0:
        shaft_hz = rpm / 60.0
        freq_res = fs / max(1, len(v))
        max_order = min(len(power) - 1, int(20 * shaft_hz / max(freq_res, 1e-12)))
        order_power = power[1:max_order] if max_order > 2 else power[1:]
    else:
        order_power = power[1:]
    order_peak_idx = int(np.argmax(order_power) + 1) if len(order_power) else peak_idx
    metrics["order_peak_ratio"] = (
        float(power[order_peak_idx] / (np.mean(order_power) + 1e-12))
        if len(order_power) and order_peak_idx < len(power)
        else 0.0
    )
    metrics["order_harmonicity"] = harmonicity(power, order_peak_idx)
    denoised = moving_average(v, 33)
    residual = v - denoised
    metrics["denoise_residual_ratio"] = float(np.mean(residual**2) / (np.mean(v.astype(np.float64) ** 2) + 1e-12))
    metrics["fault_mechanism_fidelity"] = float(
        np.log1p(metrics["kurtosis"])
        + np.log1p(metrics["crest_factor"])
        + np.log1p(metrics["envelope_peak_ratio"])
        + metrics["envelope_harmonicity"]
        + metrics["wavelet_high_energy"]
        + metrics["order_harmonicity"]
    )
    return {name: float(metrics.get(name, 0.0)) for name in FEATURE_NAMES}


def collect_features(index_csv: Path, max_items: int, seed: int) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    dataset = WindowIndexDataset(index_csv)
    indices = sample_indices(dataset.rows, max_items, seed)
    rows: list[dict[str, object]] = []
    by_view: dict[str, list[list[float]]] = {view: [] for view in VIEW_NAMES}
    for idx in indices:
        item = dataset[idx]
        x = item["x"].squeeze(0).numpy().astype(np.float32)
        metadata = item["metadata"]
        fs = float(metadata.get("target_sampling_rate", 25600) or 25600)
        rpm = parse_rpm(metadata)
        base = {
            "dataset_id": scalar(metadata.get("dataset_id", "")),
            "recording_id": scalar(metadata.get("recording_id", "")),
            "label": scalar(metadata.get("label", "")),
            "label_id": int(metadata.get("label_id", 0)),
            "condition_id": scalar(metadata.get("condition_id", "")),
            "speed_profile_id": scalar(metadata.get("speed_profile_id", "")),
            "rpm_available": int(rpm is not None),
        }
        for view in VIEW_NAMES:
            features = feature_row(x, fs, rpm, view)
            rows.append({**base, "view": view, **features})
            by_view[view].append([features[name] for name in FEATURE_NAMES])
    arrays = {view: np.asarray(values, dtype=np.float32) for view, values in by_view.items()}
    return rows, arrays


def labels_by_view(rows: list[dict[str, object]]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    labels: dict[str, list[int]] = defaultdict(list)
    domains: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        labels[str(row["view"])].append(int(row["label_id"]))
        domains[str(row["view"])].append(str(row["dataset_id"]))
    return (
        {view: np.asarray(values, dtype=np.int64) for view, values in labels.items()},
        dict(domains),
    )


def encode_domains(train_names: list[str], eval_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    domain_map = {name: idx for idx, name in enumerate(sorted(set(train_names) | set(eval_names)))}
    train_y = np.asarray([domain_map[name] for name in train_names], dtype=np.int64)
    eval_y = np.asarray([domain_map[name] for name in eval_names], dtype=np.int64)
    return train_y, eval_y, domain_map


def standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def majority_baseline(y: np.ndarray, num_classes: int) -> float:
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    return float(counts.max() / max(1, counts.sum()))


def train_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> dict[str, object]:
    import torch
    import torch.nn as nn

    if len(train_x) == 0 or len(eval_x) == 0 or num_classes <= 1:
        return {"accuracy": 0.0, "macro_f1": 0.0, "majority_baseline": majority_baseline(eval_y, num_classes)}
    torch.manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x.astype(np.float32)),
        torch.from_numpy(train_y.astype(np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    probe = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(torch.from_numpy(eval_x.astype(np.float32)).to(device)).argmax(dim=1).cpu().numpy()
    return {
        "accuracy": accuracy(eval_y, pred),
        "macro_f1": macro_f1(eval_y, pred, num_classes),
        "majority_baseline": majority_baseline(eval_y, num_classes),
    }


def fisher_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["view"])].append(row)
    out = []
    for view, group in sorted(grouped.items()):
        labels = sorted({int(row["label_id"]) for row in group})
        item: dict[str, object] = {"view": view, "num_windows": len(group)}
        for feature in FEATURE_NAMES:
            values_all = np.asarray([float(row[feature]) for row in group], dtype=np.float64)
            overall = float(values_all.mean()) if len(values_all) else 0.0
            between = 0.0
            within = 0.0
            for label in labels:
                values = np.asarray([float(row[feature]) for row in group if int(row["label_id"]) == label])
                if len(values) == 0:
                    continue
                mean = float(values.mean())
                between += len(values) * (mean - overall) ** 2
                within += float(((values - mean) ** 2).sum())
            item[f"{feature}_fisher"] = float(between / (within + 1e-12))
        fidelity_values = [float(row["fault_mechanism_fidelity"]) for row in group]
        item["fault_mechanism_fidelity_mean"] = float(np.mean(fidelity_values)) if fidelity_values else 0.0
        out.append(item)
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Intervention View Bank Diagnosis",
        "",
        "| View | Health F1 | Domain Acc. | Domain Majority | Leakage | Fidelity Fisher | Fidelity Mean | Recommendation Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['view']} | {float(row['health_macro_f1']):.6f} | "
            f"{float(row['domain_accuracy']):.6f} | {float(row['domain_majority_baseline']):.6f} | "
            f"{float(row['domain_leakage_over_majority']):.6f} | "
            f"{float(row['fault_mechanism_fidelity_fisher']):.6f} | "
            f"{float(row['fault_mechanism_fidelity_mean']):.6f} | "
            f"{float(row['recommendation_score']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Recommendation score = health Macro-F1 + 0.1 * fidelity Fisher - domain leakage over majority baseline.",
            "Use this table for source-only view selection; target test labels are not used here.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows, train_arrays = collect_features(args.train_index, args.max_train_items, args.seed)
    eval_rows, eval_arrays = collect_features(args.eval_index, args.max_eval_items, args.seed + 1)
    train_labels, train_domains = labels_by_view(train_rows)
    eval_labels, eval_domains = labels_by_view(eval_rows)
    fisher = {row["view"]: row for row in fisher_scores(eval_rows)}
    num_health_classes = int(max(max(row["label_id"] for row in train_rows), max(row["label_id"] for row in eval_rows)) + 1)

    summary_rows: list[dict[str, object]] = []
    domain_maps: dict[str, dict[str, int]] = {}
    for view in VIEW_NAMES:
        train_x, eval_x = standardize(train_arrays[view], eval_arrays[view])
        health = train_probe(
            train_x,
            train_labels[view],
            eval_x,
            eval_labels[view],
            num_classes=num_health_classes,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed,
        )
        train_domain_y, eval_domain_y, domain_map = encode_domains(train_domains[view], eval_domains[view])
        domain_maps[view] = domain_map
        domain = train_probe(
            train_x,
            train_domain_y,
            eval_x,
            eval_domain_y,
            num_classes=len(domain_map),
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed + 17,
        )
        leakage = max(0.0, float(domain["accuracy"]) - float(domain["majority_baseline"]))
        fidelity_fisher = float(fisher[view].get("fault_mechanism_fidelity_fisher", 0.0))
        recommendation = float(health["macro_f1"]) + 0.1 * fidelity_fisher - leakage
        summary_rows.append(
            {
                "target_name": args.target_name,
                "view": view,
                "health_accuracy": health["accuracy"],
                "health_macro_f1": health["macro_f1"],
                "health_majority_baseline": health["majority_baseline"],
                "domain_accuracy": domain["accuracy"],
                "domain_macro_f1": domain["macro_f1"],
                "domain_majority_baseline": domain["majority_baseline"],
                "domain_leakage_over_majority": leakage,
                "fault_mechanism_fidelity_fisher": fidelity_fisher,
                "fault_mechanism_fidelity_mean": fisher[view].get("fault_mechanism_fidelity_mean", 0.0),
                "recommendation_score": recommendation,
                "num_train_windows": len(train_arrays[view]),
                "num_eval_windows": len(eval_arrays[view]),
                "source_domains": ";".join(name for name, _ in sorted(domain_map.items(), key=lambda item: item[1])),
            }
        )

    summary_rows.sort(key=lambda row: float(row["recommendation_score"]), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, args.output_dir / f"intervention_view_bank_summary_{args.target_name}.csv")
    write_csv(fisher_scores(eval_rows), args.output_dir / f"intervention_view_bank_fisher_{args.target_name}.csv")
    write_markdown(summary_rows, args.output_dir / f"intervention_view_bank_summary_{args.target_name}.md")
    payload = {
        "target_name": args.target_name,
        "train_index": str(args.train_index),
        "eval_index": str(args.eval_index),
        "views": VIEW_NAMES,
        "features": FEATURE_NAMES,
        "domain_maps": domain_maps,
        "summary": summary_rows,
    }
    (args.output_dir / f"intervention_view_bank_summary_{args.target_name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
