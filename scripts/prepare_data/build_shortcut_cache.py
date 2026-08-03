#!/usr/bin/env python3
"""Experiment 6 (shortcut reversal): build injected view caches.

Injects a class-conditioned low-frequency tone (35/55/75 Hz, a plausible
rig/electrical artifact) into the resampled time-domain signal of every
recording, then recomputes ALL intervention views on the same canonical
window grid.

Caches built for the cross-dataset task with target=paderborn:
  sc_sources_correlated   train+val recordings (ottawa+hust), tone = class tone
  sc_target_correlated    paderborn test subsample, tone matches class
  sc_target_reversed      tone of the mirrored class (2 - y)
  sc_target_neutral       fixed 35 Hz tone for every class

Each cache dir has the same npy/master.csv layout as views_v2, so ViewCache /
CachedWindowDataset work unchanged with the original window-index CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cicman.v2 import views as V  # noqa: E402

CLASS_TONES = [35.0, 55.0, 75.0]
AMPLITUDE = 1.0  # x recording std (signals are robust-normalized, std ~ 1)
OTTAWA_CPR = 1024


def read_signal(rows: list[dict], extracted_root: Path):
    r = rows[0]
    dataset = r["dataset_id"]
    if dataset == "paderborn":
        mat_path = extracted_root / r["source_file"]
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        key = r["recording_id"] if r["recording_id"] in data else [k for k in data if not k.startswith("__")][0]
        channels = {str(ch.Name): np.asarray(ch.Data, dtype=np.float32).reshape(-1) for ch in np.atleast_1d(data[key].Y)}
        return channels[r["signal_key"]], int(r["source_sampling_rate"]), (15.0 if r["condition_id"].startswith("N09") else 25.0), None
    data = loadmat(r["source_file"])
    sig = np.asarray(data[r["signal_key"]], dtype=np.float32).reshape(-1)
    if dataset == "ottawa":
        enc = np.asarray(data[r["speed_key"]], dtype=np.float32).reshape(-1)
        return sig, int(r["source_sampling_rate"]), None, enc - enc.mean()
    shaft = float(np.asarray(data.get("fs", 0.0)).reshape(-1)[0]) if "fs" in data else 0.0
    return sig, int(r["source_sampling_rate"]), shaft, None


def process_recording(job):
    rows, extracted_root, tone_hz = job["rows"], Path(job["extracted_root"]), job["tone_hz"]
    try:
        sig, src_rate, shaft_const, enc = read_signal(rows, extracted_root)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{rows[0]['recording_id']}: {exc}"}
    sig_rs = V.robust_normalize(V.resample(sig, src_rate, V.TARGET_RATE))
    t = np.arange(len(sig_rs)) / V.TARGET_RATE
    sig_rs = (sig_rs + AMPLITUDE * np.std(sig_rs) * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32)
    if job.get("highpass_hz"):
        from scipy import signal as sps

        sos = sps.butter(4, job["highpass_hz"], btype="highpass", fs=V.TARGET_RATE, output="sos")
        sig_rs = sps.sosfiltfilt(sos, sig_rs).astype(np.float32)

    out_views = {name: [] for name in V.VIEW_SPECS}
    out_feats, meta = [], []
    for row in rows:
        start = int(row["target_start"])
        if start + V.WINDOW_LEN > len(sig_rs):
            continue
        if enc is not None:
            ca = max(0, min(start + V.WINDOW_LEN // 2 - V.ENV_CONTEXT // 2, len(sig_rs) - V.ENV_CONTEXT))
            ea, eb = int(ca / V.TARGET_RATE * src_rate), int((ca + V.ENV_CONTEXT) / V.TARGET_RATE * src_rate)
            seg = enc[ea:eb]
            above = seg > 0
            crossings = int(np.count_nonzero(above[1:] & ~above[:-1]))
            duration = max(len(seg) / src_rate, 1e-9)
            shaft = crossings / OTTAWA_CPR / duration
        else:
            shaft = float(shaft_const or 0.0)
        views, feats = V.compute_window_views(sig_rs, start, shaft)
        for name in V.VIEW_SPECS:
            out_views[name].append(views[name].astype(np.float16))
        out_feats.append(feats)
        meta.append({
            "dataset_id": row["dataset_id"],
            "recording_id": row["recording_id"],
            "window_index": int(row["window_index"]),
        })
    return {"error": None, "views": {k: np.stack(v) for k, v in out_views.items()} if meta else None,
            "feats": np.stack(out_feats) if meta else None, "meta": meta}


def build_cache(rec_rows: dict[str, list[dict]], tone_map, out_dir: Path, extracted_root: Path, workers: int, highpass_hz: float | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "master.csv").exists():
        print(f"skip existing {out_dir}")
        return
    jobs = [{"rows": rows, "extracted_root": str(extracted_root), "tone_hz": tone_map(rows), "highpass_hz": highpass_hz} for rows in rec_rows.values()]
    chunks = {name: [] for name in V.VIEW_SPECS}
    feats_all, meta_all, errors = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(process_recording, jobs, chunksize=2):
            if res.get("error"):
                errors.append(res["error"])
            elif res["views"] is not None:
                for name in V.VIEW_SPECS:
                    chunks[name].append(res["views"][name])
                feats_all.append(res["feats"])
                meta_all.extend(res["meta"])
    for name in V.VIEW_SPECS:
        np.save(out_dir / f"{name}.npy", np.concatenate(chunks[name]))
        chunks[name] = []
    np.save(out_dir / "feats.npy", np.concatenate(feats_all).astype(np.float32))
    with (out_dir / "master.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_id", "recording_id", "window_index"])
        writer.writeheader()
        writer.writerows(meta_all)
    print(f"{out_dir.name}: windows={len(meta_all)} errors={len(errors)}")
    if errors:
        print("\n".join(errors[:5]))


def load_windows(csv_path: Path):
    by_rec = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_rec[row["recording_id"]].append(row)
    return by_rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--max-target-recordings", type=int, default=300)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_dir = args.project_root / "data/paper1_cicman/cache/windows/cross_dataset_task3_source_mixed/target_dataset_paderborn"
    extracted_root = args.project_root / "data/paper1_cicman/extracted/paderborn"
    out_root = args.project_root / "data/paper1_cicman/cache/views_v2_shortcut"

    train_recs = load_windows(task_dir / "train_windows.csv")
    val_recs = load_windows(task_dir / "val_windows.csv")
    sources = {**train_recs, **val_recs}

    test_recs = load_windows(task_dir / "test_windows.csv")
    rng = np.random.default_rng(args.seed)
    keep = sorted(rng.choice(sorted(test_recs), size=min(args.max_target_recordings, len(test_recs)), replace=False).tolist())
    target = {k: test_recs[k] for k in keep}
    (out_root).mkdir(parents=True, exist_ok=True)
    (out_root / "target_recordings.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")

    def tone_correlated(rows):
        return CLASS_TONES[int(rows[0]["label_id"])]

    def tone_reversed(rows):
        return CLASS_TONES[2 - int(rows[0]["label_id"])]

    def tone_neutral(rows):
        return CLASS_TONES[0]

    build_cache(sources, tone_correlated, out_root / "sc_sources_correlated", extracted_root, args.workers)
    # intervention variant for intervention-consistent reliability estimation:
    # high-pass removes the low-frequency artifact channel counterfactually
    build_cache(sources, tone_correlated, out_root / "sc_sources_correlated_hp800", extracted_root, args.workers, highpass_hz=800.0)
    build_cache(target, tone_correlated, out_root / "sc_target_correlated", extracted_root, args.workers)
    build_cache(target, tone_reversed, out_root / "sc_target_reversed", extracted_root, args.workers)
    build_cache(target, tone_neutral, out_root / "sc_target_neutral", extracted_root, args.workers)
    print("done")


if __name__ == "__main__":
    main()
