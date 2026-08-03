#!/usr/bin/env python3
"""Pivotal probe: does the envelope ORDER spectrum transfer across datasets?

Trains a linear probe (logistic regression) on two datasets and tests on the
third (leave-one-dataset-out, task3 3-class labels), comparing three feature
views computed from identical segments:

  raw_spec   : magnitude spectrum on a fixed Hz axis (control, rig-specific)
  env_spec   : envelope spectrum on a fixed Hz axis (control)
  env_order  : envelope spectrum on a shaft-order axis (physics-normalized)

If env_order clearly beats raw_spec/env_spec on the target dataset, the
CIC-MAN v2 rebuild direction (heterogeneous intervention views with an
order-tracking agent) is validated.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import loadmat

TARGET_RATE = 25600
SEGMENT_LEN = 25600  # 1 second
SEGMENT_HOP = 25600
MAX_SEGMENTS_PER_RECORDING = 6
ORDER_GRID = np.arange(0.25, 32.0, 0.125)  # 254 bins
HZ_GRID = np.linspace(2.0, 640.0, 256)
RAW_HZ_GRID = np.linspace(10.0, 12700.0, 256)

LABELS = {0: "normal", 1: "inner", 2: "outer"}


def robust_normalize(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.std(x) or 1.0
    return ((x - med) / scale).astype(np.float32)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    g = np.gcd(src, dst)
    return sps.resample_poly(x, dst // g, src // g).astype(np.float32)


def envelope(x: np.ndarray, fs: int) -> np.ndarray:
    sos = sps.butter(4, 1000.0, btype="highpass", fs=fs, output="sos")
    xf = sps.sosfiltfilt(sos, x)
    env = np.abs(sps.hilbert(xf))
    return env - env.mean()


def amp_spectrum(x: np.ndarray, fs: int):
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, spec


def features_for_segment(seg: np.ndarray, fs: int, shaft_hz: float):
    freqs, rawspec = amp_spectrum(seg, fs)
    env = envelope(seg, fs)
    efreqs, espec = amp_spectrum(env, fs)

    raw_hz = np.interp(RAW_HZ_GRID, freqs, rawspec)
    env_hz = np.interp(HZ_GRID, efreqs, espec)
    if shaft_hz and shaft_hz > 1.0:
        env_ord = np.interp(ORDER_GRID * shaft_hz, efreqs, espec)
    else:
        env_ord = np.zeros_like(ORDER_GRID, dtype=np.float32)

    def norm(v):
        v = np.log1p(v)
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).astype(np.float32)

    return norm(raw_hz), norm(env_hz), norm(env_ord)


def estimate_shaft_hz_from_encoder(enc: np.ndarray, fs: int, cpr: int = 1024) -> float:
    """Estimate mean shaft frequency from an encoder pulse channel."""
    enc = enc - enc.mean()
    thresh = 0.0
    above = enc > thresh
    crossings = np.count_nonzero(above[1:] & ~above[:-1])
    duration = len(enc) / fs
    if duration <= 0:
        return 0.0
    return crossings / cpr / duration


def load_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def read_paderborn(extracted_root: Path, row: dict):
    mat_path = extracted_root / row["source_file"]
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    key = row["recording_id"] if row["recording_id"] in data else [k for k in data if not k.startswith("__")][0]
    struct = data[key]
    channels = {str(ch.Name): np.asarray(ch.Data, dtype=np.float32).reshape(-1) for ch in np.atleast_1d(struct.Y)}
    sig = channels["vibration_1"]
    cond = row["condition_id"]
    shaft = 15.0 if cond.startswith("N09") else 25.0
    return sig, int(row["sampling_rate"]), shaft, None


def read_ottawa(row: dict):
    data = loadmat(row["source_file"])
    sig = np.asarray(data["Channel_1"], dtype=np.float32).reshape(-1)
    enc = np.asarray(data["Channel_2"], dtype=np.float32).reshape(-1)
    return sig, int(row["sampling_rate"]), None, enc


def read_hust(row: dict):
    data = loadmat(row["source_file"])
    sig = np.asarray(data["data"], dtype=np.float32).reshape(-1)
    shaft = float(np.asarray(data.get("fs", 0.0)).reshape(-1)[0]) if "fs" in data else 0.0
    return sig, int(row["sampling_rate"]), shaft, None


def collect_dataset(name: str, manifests_dir: Path, extracted_root: Path, paderborn_per_bearing: int, rng: np.random.Generator):
    rows = load_manifest(manifests_dir / f"{name}_manifest.csv")
    rows = [r for r in rows if r["task3_label_id"] not in ("", "-1")]

    if name == "paderborn":
        by_bearing_cond = defaultdict(list)
        for r in rows:
            by_bearing_cond[(r["bearing_id"], r["condition_id"])].append(r)
        chosen = []
        per_cond = max(1, paderborn_per_bearing // 4)
        for key in sorted(by_bearing_cond):
            group = sorted(by_bearing_cond[key], key=lambda r: r["recording_id"])
            idx = rng.choice(len(group), size=min(per_cond, len(group)), replace=False)
            chosen.extend(group[i] for i in sorted(idx))
        rows = chosen

    feats = {"raw_spec": [], "env_spec": [], "env_order": []}
    labels, rec_ids, shaft_log = [], [], []
    n_fail = 0
    for r in rows:
        try:
            if name == "paderborn":
                sig, fs, shaft, enc = read_paderborn(extracted_root, r)
            elif name == "ottawa":
                sig, fs, shaft, enc = read_ottawa(r)
            else:
                sig, fs, shaft, enc = read_hust(r)
        except Exception:
            n_fail += 1
            continue

        sig_rs = robust_normalize(resample(sig, fs, TARGET_RATE))
        n_seg = min(MAX_SEGMENTS_PER_RECORDING, max(0, (len(sig_rs) - SEGMENT_LEN) // SEGMENT_HOP + 1))
        for i in range(n_seg):
            a = i * SEGMENT_HOP
            seg = sig_rs[a : a + SEGMENT_LEN]
            if enc is not None:
                ea = int(a / TARGET_RATE * fs)
                eb = int((a + SEGMENT_LEN) / TARGET_RATE * fs)
                shaft_seg = estimate_shaft_hz_from_encoder(enc[ea:eb], fs)
            else:
                shaft_seg = shaft or 0.0
            shaft_log.append(shaft_seg)
            f_raw, f_env, f_ord = features_for_segment(seg, TARGET_RATE, shaft_seg)
            feats["raw_spec"].append(f_raw)
            feats["env_spec"].append(f_env)
            feats["env_order"].append(f_ord)
            labels.append(int(r["task3_label_id"]))
            rec_ids.append(f"{name}::{r['recording_id']}")

    print(f"[{name}] recordings={len(rows)} fail={n_fail} segments={len(labels)} "
          f"shaft_hz p5/p50/p95 = {np.percentile(shaft_log, [5, 50, 95]).round(2).tolist()}")
    return {k: np.stack(v) for k, v in feats.items()}, np.array(labels), np.array(rec_ids)


def macro_f1(y_true, y_pred, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def run_probe(data, view: str, target: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_names = [n for n in data if n != target]
    Xtr = np.concatenate([data[n][0][view] for n in train_names])
    ytr = np.concatenate([data[n][1] for n in train_names])
    Xte, yte, rte = data[target][0][view], data[target][1], data[target][2]

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))

    win_f1 = macro_f1(yte, pred)
    win_acc = float(np.mean(pred == yte))

    rec_true, rec_pred = [], []
    for rec in np.unique(rte):
        m = rte == rec
        rec_true.append(yte[m][0])
        votes = np.bincount(pred[m], minlength=3)
        rec_pred.append(int(np.argmax(votes)))
    rec_f1 = macro_f1(np.array(rec_true), np.array(rec_pred))
    rec_acc = float(np.mean(np.array(rec_pred) == np.array(rec_true)))
    return {"window_acc": win_acc, "window_macro_f1": win_f1, "recording_acc": rec_acc, "recording_macro_f1": rec_f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--paderborn-per-bearing", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifests = args.project_root / "data/paper1_cicman/manifests"
    extracted = args.project_root / "data/paper1_cicman/extracted/paderborn"
    rng = np.random.default_rng(args.seed)

    data = {}
    for name in ["ottawa", "hust", "paderborn"]:
        data[name] = collect_dataset(name, manifests, extracted, args.paderborn_per_bearing, rng)

    results = {}
    for target in ["hust", "ottawa", "paderborn"]:
        for view in ["raw_spec", "env_spec", "env_order"]:
            r = run_probe(data, view, target)
            results[f"{target}/{view}"] = r
            print(f"target={target:9s} view={view:9s} "
                  f"win_acc={r['window_acc']:.3f} win_f1={r['window_macro_f1']:.3f} "
                  f"rec_acc={r['recording_acc']:.3f} rec_f1={r['recording_macro_f1']:.3f}")

    out = args.project_root / "outputs/tables/pivotal_env_order_probe.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
