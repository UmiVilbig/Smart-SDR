"""RTL-SDR hardware controller — thin wrapper around pyrtlsdr."""

import os
import sys

# On Windows (Python 3.8+) DLLs must be explicitly registered.
# Look for rtlsdr.dll next to this file, in the project root, and in PATH entries.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _dll_dir in [_ROOT, _HERE]:
    if os.path.exists(os.path.join(_dll_dir, "rtlsdr.dll")):
        os.add_dll_directory(_dll_dir)
        break

import numpy as np
from rtlsdr import RtlSdr
from contextlib import contextmanager
from typing import Optional
import threading

# Global SDR instance (one device, shared across requests)
_sdr: Optional[RtlSdr] = None
_lock = threading.Lock()


def _get_sdr() -> RtlSdr:
    global _sdr
    if _sdr is None:
        raise RuntimeError("SDR not open. Call open_device() first.")
    return _sdr


def open_device(
    sample_rate: float = 2.048e6,
    center_freq: float = 100.1e6,
    gain: float | str = "auto",
) -> dict:
    global _sdr
    with _lock:
        if _sdr is not None:
            _sdr.close()
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        sdr.gain = gain
        _sdr = sdr
    return get_status()


def close_device() -> dict:
    global _sdr
    with _lock:
        if _sdr is not None:
            _sdr.close()
            _sdr = None
    return {"status": "closed"}


def get_status() -> dict:
    with _lock:
        if _sdr is None:
            return {"open": False}
        return {
            "open": True,
            "center_freq_hz": _sdr.center_freq,
            "sample_rate_hz": _sdr.sample_rate,
            "gain": _sdr.gain,
            "freq_correction_ppm": _sdr.freq_correction,
        }


def set_frequency(freq_hz: float) -> dict:
    with _lock:
        sdr = _get_sdr()
        sdr.center_freq = freq_hz
    return get_status()


def set_gain(gain: float | str) -> dict:
    with _lock:
        sdr = _get_sdr()
        sdr.gain = gain
    return get_status()


def set_sample_rate(rate_hz: float) -> dict:
    with _lock:
        sdr = _get_sdr()
        sdr.sample_rate = rate_hz
    return get_status()


def read_samples(num_samples: int = 256 * 1024) -> np.ndarray:
    """Read IQ samples from the device."""
    with _lock:
        sdr = _get_sdr()
        samples = sdr.read_samples(num_samples)
    return samples


def get_power_spectrum(num_samples: int = 256 * 1024) -> dict:
    """Return power spectral density of the current frequency."""
    samples = read_samples(num_samples)
    sdr = _get_sdr()
    freqs = np.fft.fftfreq(len(samples), d=1.0 / sdr.sample_rate)
    freqs = np.fft.fftshift(freqs) + sdr.center_freq
    psd = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    psd_db = 10 * np.log10(psd + 1e-12)

    # Downsample for API response (return ~1024 points)
    step = max(1, len(freqs) // 1024)
    return {
        "center_freq_hz": sdr.center_freq,
        "sample_rate_hz": sdr.sample_rate,
        "freqs_hz": freqs[::step].tolist(),
        "psd_db": psd_db[::step].tolist(),
        "peak_freq_hz": float(freqs[np.argmax(psd_db)]),
        "peak_power_db": float(np.max(psd_db)),
    }


def scan_band(
    start_freq_hz: float,
    stop_freq_hz: float,
    step_hz: float = 100e3,
    dwell_samples: int = 128 * 1024,
) -> list[dict]:
    """Scan a frequency band and return peak power at each step."""
    results = []
    freq = start_freq_hz
    while freq <= stop_freq_hz:
        set_frequency(freq)
        samples = read_samples(dwell_samples)
        power_db = float(10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-12))
        results.append({"freq_hz": freq, "power_db": power_db})
        freq += step_hz
    return results
