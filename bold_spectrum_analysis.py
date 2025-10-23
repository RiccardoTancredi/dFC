#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOLD frequency analysis utilities.
- Welch PSD (robust)
- Direct FFT spectrum (more "immediate" but higher variance)
- Bandpower features in Slow-5/4/3
- Convenience plotting with canonical markers for resting-state BOLD

Usage (example):
    import numpy as np
    from bold_spectrum_analysis import (compute_psd_welch, compute_psd_fft,
                                        spectral_features, plot_spectrum, save_features_to_csv)

    TR = 0.8
    X = np.load("timeseries.npy")      # shape (T, ROIs)
    f_w, P_w = compute_psd_welch(X, TR, nperseg=256, noverlap=192)
    f_f, P_f = compute_psd_fft(X, TR)

    # Features on Welch PSD
    feats = spectral_features(f_w, P_w)
    save_features_to_csv(feats, "features_welch.csv")

    # Plot median spectrum across ROIs
    plot_spectrum(f_w, np.median(P_w, axis=0), title="Welch PSD (median across ROIs)",
                  outpath="psd_welch_median.png", log10=True, annotate_bands=True)

    plot_spectrum(f_f, np.median(P_f, axis=0), title="FFT spectrum (median across ROIs)",
                  outpath="psd_fft_median.png", log10=True, annotate_bands=True)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import welch, detrend
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple, Optional

# --------- Core PSDs ---------

def _prep_timeseries(X: np.ndarray, do_detrend: bool = True, do_zscore: bool = True) -> np.ndarray:
    Xp = X.copy()
    if do_detrend:
        Xp = detrend(Xp, axis=0, type='linear')
    if do_zscore:
        mu = Xp.mean(axis=0, keepdims=True)
        sd = Xp.std(axis=0, keepdims=True) + 1e-8
        Xp = (Xp - mu) / sd
    return Xp

def compute_psd_welch(X: np.ndarray, TR: float, nperseg: int = 256, noverlap: Optional[int] = None,
                      window: str = 'hann', scaling: str = 'density',
                      do_detrend: bool = True, do_zscore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        f: (F,) frequency axis in Hz
        Pxx: (ROIs, F) power spectral density
    """
    fs = 1.0 / TR
    if noverlap is None:
        noverlap = nperseg // 2

    Xp = _prep_timeseries(X, do_detrend, do_zscore)

    P = []
    for r in range(Xp.shape[1]):
        f, Pw = welch(Xp[:, r], fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling=scaling)
        P.append(Pw)
    Pxx = np.vstack(P)
    return f, Pxx

def compute_psd_fft(X: np.ndarray, TR: float, zero_pad_to: Optional[int] = None,
                    do_detrend: bool = True, do_zscore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct single-segment FFT (real FFT). Higher variance than Welch; useful to inspect dominant peaks quickly.

    Returns:
        f: (F,) frequency axis in Hz
        Pxx: (ROIs, F) power spectrum (density-like; scaled to be comparable up to constants)
    """
    fs = 1.0 / TR
    Xp = _prep_timeseries(X, do_detrend, do_zscore)

    T = Xp.shape[0]
    nfft = int(2 ** np.ceil(np.log2(T))) if zero_pad_to is None else int(zero_pad_to)
    # rFFT and magnitude-squared
    F = np.fft.rfftfreq(nfft, d=1.0/fs)
    P_list = []
    for r in range(Xp.shape[1]):
        Xr = Xp[:, r]
        Xr = np.pad(Xr, (0, nfft - T)) if nfft > T else Xr[:nfft]
        Xf = np.fft.rfft(Xr, n=nfft)
        P = (np.abs(Xf) ** 2) / (fs * nfft)  # simple scaling
        P_list.append(P)
    Pxx = np.vstack(P_list)
    return F, Pxx

# --------- Features ---------

def bandpower(f: np.ndarray, Pxx: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return np.zeros(Pxx.shape[0])
    return np.trapz(Pxx[:, idx], f[idx], axis=1)

def spectral_features(f: np.ndarray, Pxx: np.ndarray) -> Dict[str, np.ndarray]:
    eps = 1e-12
    # Define bands
    bands = {
        "slow5_0p01_0p027": (0.01, 0.027),
        "slow4_0p027_0p073": (0.027, 0.073),
        "slow3_0p073_0p15": (0.073, 0.15),
        "resp_like_0p2_0p4": (0.2, 0.4),
    }
    ptot = bandpower(f, Pxx, 0.01, 0.4)

    out: Dict[str, np.ndarray] = {"ptot_0p01_0p4": ptot}
    for name, (a, b) in bands.items():
        out[f"bp_{name}"] = bandpower(f, Pxx, a, b)
        out[f"frac_{name}"] = out[f"bp_{name}"] / (ptot + eps)

    # Peak frequency within 0.01–0.15
    idx = (f >= 0.01) & (f <= 0.15)
    if np.any(idx):
        f_peak = f[idx][np.argmax(Pxx[:, idx], axis=1)]
    else:
        f_peak = np.full(Pxx.shape[0], np.nan)
    out["peak_freq_0p01_0p15"] = f_peak

    # Spectral slope (log-log) in 0.01–0.15
    slopes = np.empty(Pxx.shape[0])
    slopes.fill(np.nan)
    if np.any(idx):
        fx = f[idx]
        x = np.log10(fx)
        for i in range(Pxx.shape[0]):
            y = np.log10(Pxx[i, idx] + eps)
            res = linregress(x, y)
            slopes[i] = res.slope
    out["slope_loglog_0p01_0p15"] = slopes

    return out

def save_features_to_csv(features: Dict[str, np.ndarray], out_csv: str, roi_labels: Optional[list] = None) -> None:
    keys = sorted(features.keys())
    data = {k: features[k] for k in keys}
    df = pd.DataFrame(data)
    if roi_labels is not None and len(roi_labels) == len(df):
        df.insert(0, "ROI", roi_labels)
    df.to_csv(out_csv, index=False)

# --------- Plotting ---------

def plot_spectrum(f: np.ndarray, P: np.ndarray, title: str = "", outpath: Optional[str] = None,
                  log10: bool = True, annotate_bands: bool = True, xlim: Tuple[float, float] = (0.0, 0.4)) -> None:
    """
    Plot a single spectrum curve (e.g., median across ROIs).
    - Uses a single axis and default matplotlib styles.
    - Optionally log10 on Y for readability.
    - Adds standard vertical markers and shaded bands to help interpretation.
    """
    plt.figure(figsize=(7, 4))
    y = np.log10(P + 1e-12) if log10 else P
    plt.plot(f, y, linewidth=1.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(PSD)" if log10 else "PSD")
    plt.title(title)
    plt.xlim(*xlim)

    # vertical markers
    for v in [0.01, 0.1, 0.15, 0.25, 0.30, 0.35]:
        plt.axvline(v, linestyle="--", linewidth=0.8)

    # shaded canonical bands
    if annotate_bands:
        def shade(a, b):
            plt.axvspan(a, b, alpha=0.15)
        shade(0.01, 0.027)  # Slow-5
        shade(0.027, 0.073) # Slow-4
        shade(0.073, 0.15)  # Slow-3

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=150)
    plt.close()
