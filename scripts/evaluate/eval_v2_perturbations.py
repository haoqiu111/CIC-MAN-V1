#!/usr/bin/env python3
"""Experiment 5: controlled measurement-intervention robustness for v2 models.

Perturbations are injected into the resampled time-domain signal of each
target test recording; ALL intervention views are then recomputed, so the
perturbation acts on the measurement mechanism, not on cached features.

Usage:
  python eval_v2_perturbations.py --task-dir .../target_dataset_hust \
      --checkpoint outputs/checkpoints/v2_cicman_v4_target_hust_seed42 \
      --which last --output-name v4_hust_seed42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import loadmat


def add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


add_src_to_path()

from cicman.v2 import views as V  # noqa: E402
from cicman.v2.model import CICMANv2  # noqa: E402
from recording_protocol import OFFICIAL_AGGREGATION, RecordingAccumulator  # noqa: E402

OTTAWA_CPR = 1024


def perturbations():
    rng_master = np.random.default_rng(20260705)

    def gauss(snr_db):
        def f(x, rng):
            p_sig = np.mean(x**2)
            p_noise = p_sig / (10 ** (snr_db / 10))
            return x + rng.normal(0, np.sqrt(p_noise), size=len(x)).astype(np.float32)
        return f

    def impulse(rate_hz=5.0, amp=8.0):
        def f(x, rng):
            n = int(rate_hz * len(x) / V.TARGET_RATE)
            idx = rng.integers(0, len(x), size=max(n, 1))
            y = x.copy()
            y[idx] += rng.choice([-1, 1], size=len(idx)) * amp * np.std(x)
            return y.astype(np.float32)
        return f

    def harmonic(freqs=(50.0, 150.0), amp=1.5):
        def f(x, rng):
            t = np.arange(len(x)) / V.TARGET_RATE
            y = x.copy().astype(np.float64)
            for fr in freqs:
                y += amp * np.std(x) * np.sin(2 * np.pi * fr * t + rng.uniform(0, 2 * np.pi))
            return y.astype(np.float32)
        return f

    def scale(k):
        return lambda x, rng: (k * x).astype(np.float32)

    def speed_jitter(ratio=1.03):
        def f(x, rng):
            y = V.resample(x, int(V.TARGET_RATE * 1000), int(V.TARGET_RATE * 1000 * ratio))
            if len(y) < len(x):
                y = np.pad(y, (0, len(x) - len(y)), mode="edge")
            return y[: len(x)].astype(np.float32)
        return f

    return {
        "clean": lambda x, rng: x,
        "gauss_snr10": gauss(10.0),
        "gauss_snr0": gauss(0.0),
        "impulse": impulse(),
        "harmonic": harmonic(),
        "scale_0.5": scale(0.5),
        "scale_2.0": scale(2.0),
        "speed_jitter_3pct": speed_jitter(1.03),
    }, rng_master


def read_recording_signal(rows: list[dict], extracted_root: Path):
    """Read one recording's raw signal + shaft info from its window-index rows."""
    r = rows[0]
    dataset = r["dataset_id"]
    if dataset == "paderborn":
        mat_path = extracted_root / r["source_file"]
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        key = r["recording_id"] if r["recording_id"] in data else [k for k in data if not k.startswith("__")][0]
        channels = {str(ch.Name): np.asarray(ch.Data, dtype=np.float32).reshape(-1) for ch in np.atleast_1d(data[key].Y)}
        sig = channels[r["signal_key"]]
        enc = None
        shaft = 15.0 if r["condition_id"].startswith("N09") else 25.0
    elif dataset == "ottawa":
        data = loadmat(r["source_file"])
        sig = np.asarray(data[r["signal_key"]], dtype=np.float32).reshape(-1)
        enc = np.asarray(data[r["speed_key"]], dtype=np.float32).reshape(-1)
        enc = enc - enc.mean()
        shaft = None
    else:
        data = loadmat(r["source_file"])
        sig = np.asarray(data[r["signal_key"]], dtype=np.float32).reshape(-1)
        enc = None
        shaft = float(np.asarray(data.get("fs", 0.0)).reshape(-1)[0]) if "fs" in data else 0.0
    return sig, int(r["source_sampling_rate"]), shaft, enc


def shaft_for_window(start: int, src_rate: int, shaft_const, enc, total_len: int):
    if enc is None:
        return float(shaft_const or 0.0), 0.0
    ca = max(0, min(start + V.WINDOW_LEN // 2 - V.ENV_CONTEXT // 2, total_len - V.ENV_CONTEXT))
    ea = int(ca / V.TARGET_RATE * src_rate)
    eb = int((ca + V.ENV_CONTEXT) / V.TARGET_RATE * src_rate)
    seg = enc[ea:eb]
    if len(seg) < 10:
        return 0.0, 0.0
    above = seg > 0
    crossings = int(np.count_nonzero(above[1:] & ~above[:-1]))
    duration = max(len(seg) / src_rate, 1e-9)
    shaft = crossings / OTTAWA_CPR / duration
    half = len(seg) // 2
    c1 = int(np.count_nonzero(above[1:half] & ~above[: half - 1]))
    c2 = int(np.count_nonzero(above[half + 1 :] & ~above[half:-1]))
    slope = (c2 - c1) / OTTAWA_CPR / max(duration / 2, 1e-9)
    return shaft, slope


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        rr = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * rr / (p + rr) if p + rr else 0.0)
    return float(np.mean(f1s))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--which", choices=["best", "last"], default="last")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--max-recordings", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    extracted_root = args.project_root / "data/paper1_cicman/extracted/paderborn"

    ckpt = torch.load(args.checkpoint / f"{args.which}.pt", map_location="cpu")
    model_views = ckpt["views"]
    state = ckpt["model"]
    has_prior = "router_log_prior" in state
    num_domains = state["domain_adv_head.2.weight"].shape[0]
    model = CICMANv2(
        num_classes=args.num_classes,
        views=model_views,
        num_domains=num_domains,
        router_mode="causal" if has_prior else "uniform",
        router_prior=[1.0 / len(model_views)] * len(model_views) if has_prior else None,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"state_dict: missing={missing} unexpected={unexpected}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    by_rec: dict[str, list[dict]] = defaultdict(list)
    with (args.task_dir / "test_windows.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_rec[row["recording_id"]].append(row)
    rec_ids = sorted(by_rec)
    rng = np.random.default_rng(args.seed)
    if len(rec_ids) > args.max_recordings:
        rec_ids = sorted(rng.choice(rec_ids, size=args.max_recordings, replace=False).tolist())
    print(f"recordings: {len(rec_ids)}, model views: {model_views}, which={args.which}")

    perts, _ = perturbations()
    results = {}
    for perturbation_index, (pert_name, pert_fn) in enumerate(perts.items()):
        cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        recording = RecordingAccumulator(args.num_classes)
        # Python's hash() is randomized between processes.  A positional offset
        # makes the perturbation realization reproducible across machines/runs.
        pert_rng = np.random.default_rng(args.seed + 1009 * perturbation_index)
        for rid in rec_ids:
            rows = by_rec[rid]
            try:
                sig, src_rate, shaft_const, enc = read_recording_signal(rows, extracted_root)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {rid}: {exc}")
                continue
            sig_rs = V.robust_normalize(V.resample(sig, src_rate, V.TARGET_RATE))
            sig_p = pert_fn(sig_rs, pert_rng)
            label = int(rows[0]["label_id"])

            batch_views = {v: [] for v in model_views}
            batch_feats = []
            for row in rows:
                start = int(row["target_start"])
                if start + V.WINDOW_LEN > len(sig_p):
                    continue
                shaft, slope = shaft_for_window(start, src_rate, shaft_const, enc, len(sig_p))
                views, feats = V.compute_window_views(sig_p, start, shaft, slope)
                for v in model_views:
                    batch_views[v].append(views[v])
                batch_feats.append(feats)
            if not batch_feats:
                continue
            xb = {v: torch.from_numpy(np.stack(batch_views[v])).to(device) for v in model_views}
            fb = torch.from_numpy(np.stack(batch_feats)).to(device)
            with torch.no_grad():
                out = model(xb, fb)
                probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
            preds = probs.argmax(1)
            for p_i in preds:
                cm[label, p_i] += 1
            recording.add_many(rid, label, probs)

        rec_metrics = recording.metrics(OFFICIAL_AGGREGATION)
        rec_cm = np.asarray(rec_metrics["recording_confusion_matrix"], dtype=np.int64)

        results[pert_name] = {
            "window_macro_f1": macro_f1(cm),
            "window_accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            "recording_macro_f1": macro_f1(rec_cm),
            "recording_accuracy": float(np.trace(rec_cm) / max(rec_cm.sum(), 1)),
            "num_recordings": int(rec_cm.sum()),
            "recording_aggregation": OFFICIAL_AGGREGATION,
            "recording_vote_ties": int(rec_metrics["vote_ties"]),
            "recording_probability_tiebreaks": int(rec_metrics["probability_tiebreaks"]),
        }
        r = results[pert_name]
        print(f"{pert_name:18s} win_f1={r['window_macro_f1']:.4f} rec_f1={r['recording_macro_f1']:.4f}", flush=True)

    out_dir = args.project_root / "outputs/tables/v2_perturbations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.output_name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"-> {out_dir / (args.output_name + '.json')}")


if __name__ == "__main__":
    main()
