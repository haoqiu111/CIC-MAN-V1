#!/usr/bin/env python3
"""Build the global precomputed multi-view window cache for CIC-MAN v2.

For every recording in the three manifests, this script resamples the signal
to 25.6 kHz, robust-normalizes it per recording, cuts the canonical window
grid (4096 samples, hop 2048 - identical to the existing split/window
indexes), and stores six intervention views plus quality/fidelity features:

  raw        (N, 4096)  f16   normalized waveform
  denoise    (N, 4096)  f16   spectral soft-threshold denoised waveform
  env_spec   (N, 256)   f16   envelope spectrum, fixed Hz axis 2-640 Hz
  env_order  (N, 256)   f16   envelope spectrum, shaft-order axis 0.25-32 ord
  stft       (N, 64, 32) f16  log-magnitude STFT
  cwt        (N, 32, 32) f16  log-magnitude Morlet scalogram
  feats      (N, 16)    f32   quality + data-driven fault-mechanism fidelity
  master.csv row-aligned metadata (dataset_id, recording_id, window_index,
              labels, domain fields, shaft_hz)

Envelope views use a longer centered context (16384 samples) for usable
frequency resolution; the context never crosses recording boundaries, so the
recording-level no-leakage guarantee is unchanged.

Shaft frequency per window: Paderborn nominal from condition code
(N09=15 Hz, N15=25 Hz), Ottawa from the encoder channel (1024 CPR pulse
count over the window context), HUST from the `fs` scalar field (shaft
frequency proxy, 22-25 Hz).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import loadmat
from scipy.stats import kurtosis as sp_kurtosis

TARGET_RATE = 25600
WINDOW_LEN = 4096
HOP_LEN = 2048
ENV_CONTEXT = 16384
ORDER_GRID = np.linspace(0.25, 32.0, 256).astype(np.float64)
HZ_GRID = np.linspace(2.0, 640.0, 256).astype(np.float64)
OTTAWA_CPR = 1024

VIEW_SPECS = {
    "raw": (WINDOW_LEN,),
    "denoise": (WINDOW_LEN,),
    "env_spec": (256,),
    "env_order": (256,),
    "stft": (64, 32),
    "cwt": (32, 32),
}
FEAT_DIM = 16

CWT_FREQS = np.geomspace(50.0, 8000.0, 32)
CWT_W = 6.0
CWT_WIDTHS = CWT_W * TARGET_RATE / (2 * np.pi * CWT_FREQS)

# Generic characteristic-order bands (data-driven fidelity operator; no
# bearing-geometry lookup so the pipeline stays target-free/geometry-agnostic).
BAND_CAGE = (0.30, 0.55)
BAND_BSF = (1.8, 2.7)
BAND_BPFO = (2.7, 4.5)
BAND_BPFI = (4.5, 6.8)


def robust_normalize(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(x)) or 1.0
    return ((x - med) / scale).astype(np.float32)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    g = np.gcd(src, dst)
    return sps.resample_poly(x, dst // g, src // g).astype(np.float32)


_HP_SOS = sps.butter(4, 1000.0, btype="highpass", fs=TARGET_RATE, output="sos")


def spectral_denoise(w: np.ndarray) -> np.ndarray:
    spec = np.fft.rfft(w)
    mag = np.abs(spec)
    thr = 2.0 * np.median(mag)
    new_mag = np.maximum(mag - thr, 0.0)
    return np.fft.irfft(spec * (new_mag / np.maximum(mag, 1e-12)), n=len(w)).astype(np.float32)


def log_unit(v: np.ndarray) -> np.ndarray:
    v = np.log1p(np.maximum(v, 0.0))
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


def compute_stft(w: np.ndarray) -> np.ndarray:
    _, _, Z = sps.stft(w, fs=TARGET_RATE, nperseg=256, noverlap=128, padded=False, boundary=None)
    mag = np.log1p(np.abs(Z))  # (129, frames~31)
    mag = mag[:128].reshape(64, 2, -1).mean(axis=1)
    frames = mag.shape[1]
    if frames < 32:
        mag = np.pad(mag, ((0, 0), (0, 32 - frames)), mode="edge")
    else:
        mag = mag[:, :32]
    m, s = mag.mean(), mag.std() or 1.0
    return ((mag - m) / s).astype(np.float32)


def compute_cwt(w: np.ndarray) -> np.ndarray:
    out = np.empty((32, 32), dtype=np.float32)
    for i, width in enumerate(CWT_WIDTHS):
        length = min(int(10 * width), WINDOW_LEN)
        if hasattr(sps, "morlet2"):
            wav = sps.morlet2(length, width, w=CWT_W)
        else:
            # SciPy >=1.15 removed morlet2. This is its documented normalized
            # complex-Morlet definition, retained for cache reproducibility.
            x = (np.arange(length) - (length - 1.0) / 2.0) / width
            wav = (np.pi ** -0.25) * np.sqrt(1.0 / width) * np.exp(1j * CWT_W * x) * np.exp(-0.5 * x**2)
        conv = sps.fftconvolve(w, wav, mode="same")
        mag = np.log1p(np.abs(conv))
        out[i] = mag.reshape(32, -1).mean(axis=1)
    m, s = out.mean(), out.std() or 1.0
    return ((out - m) / s).astype(np.float32)


def band_energy(orders: np.ndarray, spec: np.ndarray, band: tuple[float, float]) -> float:
    m = (orders >= band[0]) & (orders <= band[1])
    return float(np.sum(spec[m] ** 2))


def process_recording(job: dict):
    """Worker: returns (meta_rows, {view: array}) for one recording."""
    row = job["row"]
    dataset = row["dataset_id"]
    try:
        if dataset == "paderborn":
            mat_path = Path(job["extracted_root"]) / row["source_file"]
            data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            key = row["recording_id"] if row["recording_id"] in data else [k for k in data if not k.startswith("__")][0]
            channels = {str(ch.Name): np.asarray(ch.Data, dtype=np.float32).reshape(-1) for ch in np.atleast_1d(data[key].Y)}
            sig = channels[row["signal_key"]]
            enc = None
            shaft_const = 15.0 if row["condition_id"].startswith("N09") else 25.0
        elif dataset == "ottawa":
            data = loadmat(row["source_file"])
            sig = np.asarray(data["Channel_1"], dtype=np.float32).reshape(-1)
            enc = np.asarray(data["Channel_2"], dtype=np.float32).reshape(-1)
            enc = enc - enc.mean()
            shaft_const = None
        elif dataset == "hust":
            data = loadmat(row["source_file"])
            sig = np.asarray(data["data"], dtype=np.float32).reshape(-1)
            enc = None
            shaft_const = float(np.asarray(data.get("fs", 0.0)).reshape(-1)[0]) if "fs" in data else 0.0
        else:
            raise ValueError(f"unsupported dataset: {dataset}")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{dataset}/{row['recording_id']}: {exc}"}

    src_rate = int(row["sampling_rate"])
    sig_rs = robust_normalize(resample(sig, src_rate, TARGET_RATE))
    if job.get("highpass_hz"):
        hp_sos = sps.butter(4, float(job["highpass_hz"]), btype="highpass", fs=TARGET_RATE, output="sos")
        sig_rs = sps.sosfiltfilt(hp_sos, sig_rs).astype(np.float32)
    n_win = (len(sig_rs) - WINDOW_LEN) // HOP_LEN + 1 if len(sig_rs) >= WINDOW_LEN else 0
    if n_win <= 0:
        return {"error": None, "meta": [], "views": None}

    views = {name: np.empty((n_win,) + shape, dtype=np.float16) for name, shape in VIEW_SPECS.items()}
    feats = np.empty((n_win, FEAT_DIM), dtype=np.float32)
    meta = []

    total_len = len(sig_rs)
    for wi in range(n_win):
        a = wi * HOP_LEN
        b = a + WINDOW_LEN
        w = sig_rs[a:b]

        # context for envelope views
        ca = max(0, min(a + WINDOW_LEN // 2 - ENV_CONTEXT // 2, total_len - ENV_CONTEXT))
        cb = ca + ENV_CONTEXT
        ctx = sig_rs[ca:cb]

        # shaft frequency
        if enc is not None:
            ea = int(ca / TARGET_RATE * src_rate)
            eb = int(cb / TARGET_RATE * src_rate)
            seg = enc[ea:eb]
            above = seg > 0
            crossings = int(np.count_nonzero(above[1:] & ~above[:-1]))
            duration = max(len(seg) / src_rate, 1e-9)
            shaft_hz = crossings / OTTAWA_CPR / duration
            # speed slope: compare halves
            half = len(seg) // 2
            c1 = int(np.count_nonzero(above[1:half] & ~above[: half - 1]))
            c2 = int(np.count_nonzero(above[half + 1 :] & ~above[half:-1]))
            slope = (c2 - c1) / OTTAWA_CPR / max(duration / 2, 1e-9)
        else:
            shaft_hz = float(shaft_const or 0.0)
            slope = 0.0

        # envelope on context
        ctx_f = sps.sosfiltfilt(_HP_SOS, ctx)
        env = np.abs(sps.hilbert(ctx_f))
        env = env - env.mean()
        espec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
        efreqs = np.fft.rfftfreq(len(env), 1.0 / TARGET_RATE)

        env_hz = np.interp(HZ_GRID, efreqs, espec)
        if shaft_hz > 1.0:
            env_ord = np.interp(ORDER_GRID * shaft_hz, efreqs, espec)
        else:
            env_ord = np.zeros_like(ORDER_GRID)

        views["raw"][wi] = w.astype(np.float16)
        views["denoise"][wi] = spectral_denoise(w).astype(np.float16)
        views["env_spec"][wi] = log_unit(env_hz).astype(np.float16)
        views["env_order"][wi] = log_unit(env_ord).astype(np.float16)
        views["stft"][wi] = compute_stft(w).astype(np.float16)
        views["cwt"][wi] = compute_cwt(w).astype(np.float16)

        # quality + fidelity features
        rms = float(np.sqrt(np.mean(w**2)))
        peak = float(np.max(np.abs(w)))
        kurt = float(sp_kurtosis(w))
        crest = peak / (rms + 1e-9)
        wspec = np.abs(np.fft.rfft(w))
        p = wspec / (np.sum(wspec) + 1e-12)
        spec_entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))
        env_kurt = float(sp_kurtosis(env))
        total_e = float(np.sum(np.interp(ORDER_GRID, ORDER_GRID, env_ord) ** 2)) + 1e-12 if shaft_hz > 1.0 else 1.0
        e_cage = band_energy(ORDER_GRID, env_ord, BAND_CAGE) / total_e if shaft_hz > 1.0 else 0.0
        e_bsf = band_energy(ORDER_GRID, env_ord, BAND_BSF) / total_e if shaft_hz > 1.0 else 0.0
        e_bpfo = band_energy(ORDER_GRID, env_ord, BAND_BPFO) / total_e if shaft_hz > 1.0 else 0.0
        e_bpfi = band_energy(ORDER_GRID, env_ord, BAND_BPFI) / total_e if shaft_hz > 1.0 else 0.0
        e_1x = band_energy(ORDER_GRID, env_ord, (0.85, 1.15)) / total_e if shaft_hz > 1.0 else 0.0
        peak_order = float(ORDER_GRID[int(np.argmax(env_ord))]) if shaft_hz > 1.0 else 0.0
        feats[wi] = np.array(
            [rms, peak, kurt, crest, spec_entropy, env_kurt, shaft_hz / 30.0, slope,
             e_cage, e_bsf, e_bpfo, e_bpfi, e_1x, peak_order / 32.0,
             float(n_win), float(wi) / max(n_win - 1, 1)],
            dtype=np.float32,
        )

        meta.append({
            "dataset_id": dataset,
            "recording_id": row["recording_id"],
            "window_index": wi,
            "target_start": a,
            "fault_label_id": row.get("fault_label_id", ""),
            "task3_label_id": row.get("task3_label_id", ""),
            "task4_label_id": row.get("task4_label_id", ""),
            "is_compound": row.get("is_compound", "0"),
            "bearing_id": row.get("bearing_id", ""),
            "bearing_type_id": row.get("bearing_type_id", ""),
            "condition_id": row.get("condition_id", ""),
            "speed_profile_id": row.get("speed_profile_id", ""),
            "trial_id": row.get("trial_id", ""),
            "shaft_hz": f"{shaft_hz:.3f}",
        })

    return {"error": None, "meta": meta, "views": views, "feats": feats}


def load_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-name", default="views_v2")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-recordings", type=int, default=None, help="Debug limit per dataset.")
    parser.add_argument("--highpass-hz", type=float, default=None,
                        help="Counterfactual measurement intervention: high-pass the normalized signal before view computation.")
    parser.add_argument("--datasets", default="ottawa,hust,paderborn",
                        help="Comma-separated manifest stems to include.")
    args = parser.parse_args()

    manifests_dir = args.project_root / "data/paper1_cicman/manifests"
    extracted_root = args.project_root / "data/paper1_cicman/extracted/paderborn"
    out_dir = args.project_root / "data/paper1_cicman/cache" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for name in args.datasets.split(","):
        rows = load_manifest(manifests_dir / f"{name}_manifest.csv")
        if args.max_recordings:
            rows = rows[: args.max_recordings]
        for row in rows:
            jobs.append({"row": row, "extracted_root": str(extracted_root), "highpass_hz": args.highpass_hz})
    print(f"total recordings: {len(jobs)}")

    meta_all: list[dict] = []
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_SPECS}
    feat_chunks: list[np.ndarray] = []
    errors: list[str] = []

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_recording, jobs, chunksize=4):
            done += 1
            if result.get("error"):
                errors.append(result["error"])
            elif result["views"] is not None:
                meta_all.extend(result["meta"])
                for name in VIEW_SPECS:
                    chunks[name].append(result["views"][name])
                feat_chunks.append(result["feats"])
            if done % 100 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(jobs)} recordings ({rate:.1f}/s, windows={len(meta_all)}, errors={len(errors)})", flush=True)

    n = len(meta_all)
    print(f"writing cache: {n} windows -> {out_dir}")
    for name, shape in VIEW_SPECS.items():
        arr = np.concatenate(chunks[name], axis=0)
        assert arr.shape == (n,) + shape, f"{name}: {arr.shape}"
        np.save(out_dir / f"{name}.npy", arr)
        chunks[name] = []
    np.save(out_dir / "feats.npy", np.concatenate(feat_chunks, axis=0))

    with (out_dir / "master.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_all[0].keys()))
        writer.writeheader()
        writer.writerows(meta_all)

    report = {
        "num_windows": n,
        "num_recordings": len(jobs),
        "errors": errors,
        "views": {k: list(v) for k, v in VIEW_SPECS.items()},
        "feat_dim": FEAT_DIM,
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"done. errors={len(errors)}")
    if errors:
        print("\n".join(errors[:10]))


if __name__ == "__main__":
    main()
