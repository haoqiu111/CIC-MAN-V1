"""Per-window intervention-view computation shared by the cache builder and
on-the-fly evaluation (perturbation robustness, shortcut reversal).

Numerics must stay identical to the views_v2 cache build.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis as sp_kurtosis

TARGET_RATE = 25600
WINDOW_LEN = 4096
HOP_LEN = 2048
ENV_CONTEXT = 16384
ORDER_GRID = np.linspace(0.25, 32.0, 256).astype(np.float64)
HZ_GRID = np.linspace(2.0, 640.0, 256).astype(np.float64)

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

BAND_CAGE = (0.30, 0.55)
BAND_BSF = (1.8, 2.7)
BAND_BPFO = (2.7, 4.5)
BAND_BPFI = (4.5, 6.8)

_HP_SOS = sps.butter(4, 1000.0, btype="highpass", fs=TARGET_RATE, output="sos")


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
    g = np.gcd(int(src), int(dst))
    return sps.resample_poly(x, dst // g, src // g).astype(np.float32)


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
    mag = np.log1p(np.abs(Z))
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
            # scipy.signal.morlet2 was removed in SciPy 1.15.  This is its
            # former definition, retained verbatim at the formula level so
            # view generation stays checkpoint-compatible.
            x = (np.arange(length, dtype=np.float64) - (length - 1.0) / 2.0) / width
            wav = (np.pi ** -0.25) * np.exp(1j * CWT_W * x) * np.exp(-0.5 * x * x) / np.sqrt(width)
        conv = sps.fftconvolve(w, wav, mode="same")
        mag = np.log1p(np.abs(conv))
        out[i] = mag.reshape(32, -1).mean(axis=1)
    m, s = out.mean(), out.std() or 1.0
    return ((out - m) / s).astype(np.float32)


def band_energy(orders: np.ndarray, spec: np.ndarray, band: tuple[float, float]) -> float:
    m = (orders >= band[0]) & (orders <= band[1])
    return float(np.sum(spec[m] ** 2))


def compute_window_views(sig_rs: np.ndarray, start: int, shaft_hz: float, slope: float = 0.0):
    """Compute all views + feature vector for the window starting at `start`
    within the resampled, per-recording-normalized signal `sig_rs`."""
    total_len = len(sig_rs)
    a, b = start, start + WINDOW_LEN
    w = sig_rs[a:b]
    n_win = (total_len - WINDOW_LEN) // HOP_LEN + 1
    wi = start // HOP_LEN

    ca = max(0, min(a + WINDOW_LEN // 2 - ENV_CONTEXT // 2, total_len - ENV_CONTEXT))
    ctx = sig_rs[ca : ca + ENV_CONTEXT]

    ctx_f = sps.sosfiltfilt(_HP_SOS, ctx)
    env = np.abs(sps.hilbert(ctx_f))
    env = env - env.mean()
    espec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    efreqs = np.fft.rfftfreq(len(env), 1.0 / TARGET_RATE)

    env_hz = np.interp(HZ_GRID, efreqs, espec)
    env_ord = np.interp(ORDER_GRID * shaft_hz, efreqs, espec) if shaft_hz > 1.0 else np.zeros_like(ORDER_GRID)

    views = {
        "raw": w.astype(np.float32),
        "denoise": spectral_denoise(w),
        "env_spec": log_unit(env_hz),
        "env_order": log_unit(env_ord),
        "stft": compute_stft(w),
        "cwt": compute_cwt(w),
    }

    rms = float(np.sqrt(np.mean(w**2)))
    peak = float(np.max(np.abs(w)))
    kurt = float(sp_kurtosis(w))
    crest = peak / (rms + 1e-9)
    wspec = np.abs(np.fft.rfft(w))
    p = wspec / (np.sum(wspec) + 1e-12)
    spec_entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))
    env_kurt = float(sp_kurtosis(env))
    total_e = float(np.sum(env_ord**2)) + 1e-12 if shaft_hz > 1.0 else 1.0
    e = {
        "cage": band_energy(ORDER_GRID, env_ord, BAND_CAGE) / total_e if shaft_hz > 1.0 else 0.0,
        "bsf": band_energy(ORDER_GRID, env_ord, BAND_BSF) / total_e if shaft_hz > 1.0 else 0.0,
        "bpfo": band_energy(ORDER_GRID, env_ord, BAND_BPFO) / total_e if shaft_hz > 1.0 else 0.0,
        "bpfi": band_energy(ORDER_GRID, env_ord, BAND_BPFI) / total_e if shaft_hz > 1.0 else 0.0,
        "one_x": band_energy(ORDER_GRID, env_ord, (0.85, 1.15)) / total_e if shaft_hz > 1.0 else 0.0,
    }
    peak_order = float(ORDER_GRID[int(np.argmax(env_ord))]) if shaft_hz > 1.0 else 0.0
    feats = np.array(
        [rms, peak, kurt, crest, spec_entropy, env_kurt, shaft_hz / 30.0, slope,
         e["cage"], e["bsf"], e["bpfo"], e["bpfi"], e["one_x"], peak_order / 32.0,
         float(n_win), float(wi) / max(n_win - 1, 1)],
        dtype=np.float32,
    )
    return views, feats
