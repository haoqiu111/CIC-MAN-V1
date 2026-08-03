#!/usr/bin/env python3
"""Experiment 7: fault-mechanism fidelity via controlled fault injection.

Known synthetic fault signatures are injected into REAL healthy recordings of
the target dataset; the model evaluated is the one trained WITHOUT that
dataset (target-free). Signature = repetitive impacts at a characteristic
order (damped 3 kHz resonance bursts); BPFI adds 1x-shaft amplitude
modulation.

Reported per view/model:
  - characteristic-order localization error of the env_order view
  - band-energy preservation of the denoise path (fidelity of denoising)
  - diagnosis consistency: does the target-free model classify the injected
    fault correctly, as a function of injection strength?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


add_src_to_path()

from cicman.v2 import views as V  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402

INJECTIONS = {
    "bpfo": {"order": 3.5, "modulate": False, "expected_class": 2},
    "bpfi": {"order": 5.4, "modulate": True, "expected_class": 1},
    "bsf": {"order": 2.4, "modulate": False, "expected_class": None},
}
LEVELS = [0.3, 0.6, 1.0]
RESONANCE_HZ = 3000.0
TAU = 0.0012
OTTAWA_CPR = 1024


def synth_fault(n: int, shaft_hz: float, order: float, modulate: bool, rng) -> np.ndarray:
    fs = V.TARGET_RATE
    f_c = order * shaft_hz
    if f_c <= 0:
        return np.zeros(n, dtype=np.float32)
    t = np.arange(n) / fs
    out = np.zeros(n, dtype=np.float64)
    burst_len = int(6 * TAU * fs)
    bt = np.arange(burst_len) / fs
    burst = np.exp(-bt / TAU) * np.sin(2 * np.pi * RESONANCE_HZ * bt)
    period = fs / f_c
    k = 0
    while True:
        pos = int(k * period + rng.uniform(-0.01, 0.01) * period)
        k += 1
        if pos >= n:
            break
        if pos < 0:
            continue
        amp = 1.0
        if modulate:
            amp = 1.0 + 0.6 * np.sin(2 * np.pi * shaft_hz * pos / fs)
        end = min(pos + burst_len, n)
        if end > pos:
            out[pos:end] += amp * burst[: end - pos]
    return out.astype(np.float32)


def read_signal(row: dict, extracted_root: Path):
    dataset = row["dataset_id"]
    if dataset == "paderborn":
        mat_path = extracted_root / row["source_file"]
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        key = row["recording_id"] if row["recording_id"] in data else [k for k in data if not k.startswith("__")][0]
        channels = {str(ch.Name): np.asarray(ch.Data, dtype=np.float32).reshape(-1) for ch in np.atleast_1d(data[key].Y)}
        return channels[row["signal_key"]], int(row["sampling_rate"]), (15.0 if row["condition_id"].startswith("N09") else 25.0), None
    data = loadmat(row["source_file"])
    sig = np.asarray(data[row["signal_key"]], dtype=np.float32).reshape(-1)
    if dataset == "ottawa":
        enc = np.asarray(data[row["speed_key"]], dtype=np.float32).reshape(-1)
        return sig, int(row["sampling_rate"]), None, enc - enc.mean()
    shaft = float(np.asarray(data.get("fs", 0.0)).reshape(-1)[0]) if "fs" in data else 0.0
    return sig, int(row["sampling_rate"]), shaft, None


def window_shaft(start, src_rate, shaft_const, enc, total_len):
    if enc is None:
        return float(shaft_const or 0.0)
    ca = max(0, min(start + V.WINDOW_LEN // 2 - V.ENV_CONTEXT // 2, total_len - V.ENV_CONTEXT))
    ea, eb = int(ca / V.TARGET_RATE * src_rate), int((ca + V.ENV_CONTEXT) / V.TARGET_RATE * src_rate)
    seg = enc[ea:eb]
    if len(seg) < 10:
        return 0.0
    above = seg > 0
    crossings = int(np.count_nonzero(above[1:] & ~above[:-1]))
    return crossings / OTTAWA_CPR / max(len(seg) / src_rate, 1e-9)


def env_order_peak(view_env_order: np.ndarray, lo=1.5, hi=8.0):
    grid = V.ORDER_GRID
    m = (grid >= lo) & (grid <= hi)
    return float(grid[m][int(np.argmax(view_env_order[m]))])


def band_ratio(sig_window: np.ndarray, shaft_hz: float, order: float) -> float:
    """Energy near the injected order in the envelope spectrum of a waveform."""
    from scipy import signal as sps

    if shaft_hz <= 0:
        return 0.0
    xf = sps.sosfiltfilt(V._HP_SOS, sig_window)
    env = np.abs(sps.hilbert(xf))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), 1.0 / V.TARGET_RATE)
    f_c = order * shaft_hz
    band = (freqs >= f_c - 3) & (freqs <= f_c + 3)
    total = (freqs >= 2) & (freqs <= 640)
    return float(np.sum(spec[band] ** 2) / max(np.sum(spec[total] ** 2), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--per-dataset", type=int, default=8, help="healthy recordings per target dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    root = args.project_root
    extracted_root = root / "data/paper1_cicman/extracted/paderborn"
    manifests = root / "data/paper1_cicman/manifests"
    ckpt_root = root / "outputs/checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    rows_out = []
    for target in ["hust", "ottawa", "paderborn"]:
        ckpt = torch.load(ckpt_root / f"v2_cicman_v4_target_{target}_seed{args.checkpoint_seed}" / "last.pt", map_location="cpu")
        views_list = ckpt["views"]
        state = ckpt["model"]
        model = CICMANv2(num_classes=3, views=views_list,
                         num_domains=state["domain_adv_head.2.weight"].shape[0],
                         router_mode="causal", router_prior=[1.0 / len(views_list)] * len(views_list))
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

        with (manifests / f"{target}_manifest.csv").open(newline="", encoding="utf-8") as f:
            healthy = [r for r in csv.DictReader(f) if r["task3_label_id"] == "0"]
        idx = rng.choice(len(healthy), size=min(args.per_dataset, len(healthy)), replace=False)
        chosen = [healthy[i] for i in sorted(idx)]

        for rec in chosen:
            try:
                sig, src_rate, shaft_const, enc = read_signal(rec, extracted_root)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {rec['recording_id']}: {exc}")
                continue
            sig_rs = V.robust_normalize(V.resample(sig, src_rate, V.TARGET_RATE))
            n_win = min(8, (len(sig_rs) - V.WINDOW_LEN) // V.HOP_LEN + 1)
            if n_win <= 0:
                continue

            for inj_name, inj in INJECTIONS.items():
                for level in LEVELS:
                    mean_shaft = float(shaft_const) if enc is None else window_shaft(0, src_rate, None, enc, len(sig_rs))
                    fault = synth_fault(len(sig_rs), mean_shaft, inj["order"], inj["modulate"], rng)
                    sig_inj = sig_rs + level * np.std(sig_rs) * fault / max(np.std(fault), 1e-9)

                    batch_v = {v: [] for v in views_list}
                    batch_f, loc_errs, ratios_raw, ratios_den = [], [], [], []
                    for wi in range(n_win):
                        start = wi * V.HOP_LEN
                        shaft = window_shaft(start, src_rate, shaft_const, enc, len(sig_inj)) if enc is not None else mean_shaft
                        vw, ft = V.compute_window_views(sig_inj, start, shaft)
                        for v in views_list:
                            batch_v[v].append(vw[v])
                        batch_f.append(ft)
                        if shaft > 1:
                            loc_errs.append(abs(env_order_peak(vw["env_order"]) - inj["order"]))
                            w = sig_inj[start : start + V.WINDOW_LEN]
                            ratios_raw.append(band_ratio(w, shaft, inj["order"]))
                            ratios_den.append(band_ratio(vw["denoise"], shaft, inj["order"]))
                    xb = {v: torch.from_numpy(np.stack(batch_v[v])).to(device) for v in views_list}
                    fb = torch.from_numpy(np.stack(batch_f)).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(model(xb, fb)["logits"], dim=1).cpu().numpy()
                    pred = int(np.argmax(probs.sum(0)))

                    rows_out.append({
                        "target": target, "recording": rec["recording_id"], "injection": inj_name,
                        "level": level, "shaft_hz": round(mean_shaft, 2),
                        "order_loc_error": round(float(np.mean(loc_errs)), 4) if loc_errs else "",
                        "order_hit_rate": round(float(np.mean([e < 0.3 for e in loc_errs])), 4) if loc_errs else "",
                        "band_ratio_raw": round(float(np.mean(ratios_raw)), 5) if ratios_raw else "",
                        "band_ratio_denoise": round(float(np.mean(ratios_den)), 5) if ratios_den else "",
                        "model_pred": pred,
                        "pred_correct": int(pred == inj["expected_class"]) if inj["expected_class"] is not None else "",
                    })
        print(f"{target}: done ({len(chosen)} healthy recordings)", flush=True)

    out_dir = root / "outputs/tables"
    csv_path = out_dir / "v2_fidelity_injection_detail.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    # summary
    import pandas as pd

    df = pd.read_csv(csv_path)
    md = out_dir / "v2_fidelity_injection_summary.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Experiment 7: Fault-Mechanism Fidelity (injection into target healthy recordings)\n\n")
        f.write("Target-free cicman_v4 checkpoints; damped-resonance impact trains at known characteristic orders.\n\n")
        f.write("## Order localization / denoise energy preservation / diagnosis consistency\n\n")
        f.write("| Target | Injection | Level | Loc. err (ord) | Hit<0.3ord | Band ratio raw | Band ratio denoise | Pred acc |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        g = df.groupby(["target", "injection", "level"]).agg(
            loc=("order_loc_error", "mean"), hit=("order_hit_rate", "mean"),
            br=("band_ratio_raw", "mean"), bd=("band_ratio_denoise", "mean"),
            acc=("pred_correct", "mean"))
        for (t, i, l), r in g.iterrows():
            acc = f"{r['acc']:.3f}" if not np.isnan(r["acc"]) else "-"
            f.write(f"| {t} | {i} | {l} | {r['loc']:.3f} | {r['hit']:.3f} | {r['br']:.5f} | {r['bd']:.5f} | {acc} |\n")
    print(f"-> {md}")


if __name__ == "__main__":
    main()
