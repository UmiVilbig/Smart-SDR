"""FM demodulation and basic audio analysis using numpy/scipy."""

import numpy as np
from scipy.signal import butter, lfilter, resample_poly
from math import gcd
from typing import Optional
import wave
import io
import base64


def fm_demodulate(
    iq_samples: np.ndarray,
    sample_rate: float = 2.048e6,
    audio_rate: int = 44100,
    deemphasis_tau: float = 75e-6,
) -> np.ndarray:
    """
    Demodulate wideband FM to audio samples.

    Returns float32 audio at `audio_rate` Hz, range [-1, 1].
    """
    # ── 1. FM discriminator (differentiate phase) ──────────────────────────
    # x[n] = conj(x[n-1]) * x[n] → angle gives instantaneous frequency
    diff = iq_samples[1:] * np.conj(iq_samples[:-1])
    demod = np.angle(diff)

    # ── 2. De-emphasis filter (75 µs for NA/Japan, 50 µs for Europe) ───────
    dt = 1.0 / sample_rate
    alpha = dt / (deemphasis_tau + dt)
    b_de = [alpha]
    a_de = [1, -(1 - alpha)]
    demod = lfilter(b_de, a_de, demod)

    # ── 3. Downsample to audio rate ─────────────────────────────────────────
    g = gcd(int(audio_rate), int(sample_rate))
    up = int(audio_rate) // g
    down = int(sample_rate) // g
    audio = resample_poly(demod, up, down)

    # ── 4. Normalise ────────────────────────────────────────────────────────
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


def audio_to_wav_b64(audio: np.ndarray, sample_rate: int = 44100) -> str:
    """Encode float32 audio as a base64-encoded WAV string."""
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def estimate_signal_quality(iq_samples: np.ndarray) -> dict:
    """Return basic signal quality metrics from IQ samples."""
    power = np.mean(np.abs(iq_samples) ** 2)
    diff = iq_samples[1:] * np.conj(iq_samples[:-1])
    demod = np.angle(diff)
    signal_power = np.mean(demod ** 2)
    noise_floor = np.percentile(np.abs(demod), 10) ** 2
    snr_db = float(10 * np.log10(signal_power / (noise_floor + 1e-12)))
    return {
        "power_db": float(10 * np.log10(power + 1e-12)),
        "snr_db": snr_db,
        "dc_offset_i": float(np.mean(iq_samples.real)),
        "dc_offset_q": float(np.mean(iq_samples.imag)),
    }


def demodulate_fm_station(
    iq_samples: np.ndarray,
    sample_rate: float = 2.048e6,
    audio_rate: int = 44100,
    include_audio: bool = False,
) -> dict:
    """
    Full pipeline: IQ → demod → quality metrics → optional WAV.
    Returns a dict safe to return from a REST endpoint.
    """
    quality = estimate_signal_quality(iq_samples)
    audio = fm_demodulate(iq_samples, sample_rate, audio_rate)

    result = {
        **quality,
        "audio_samples": len(audio),
        "audio_rate_hz": audio_rate,
        "duration_sec": round(len(audio) / audio_rate, 2),
    }
    if include_audio:
        result["wav_b64"] = audio_to_wav_b64(audio, audio_rate)
    return result
