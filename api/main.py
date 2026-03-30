"""
RTL-SDR REST API
================
Run:  uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np

from api import sdr_controller as sdr
from api.demodulation import demodulate_fm_station

app = FastAPI(title="RTL-SDR REST API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────

class OpenRequest(BaseModel):
    sample_rate: float = 2.048e6
    center_freq: float = 100.1e6
    gain: float | str = "auto"

class FrequencyRequest(BaseModel):
    freq_hz: float

class GainRequest(BaseModel):
    gain: float | str

class SampleRateRequest(BaseModel):
    rate_hz: float

class ScanRequest(BaseModel):
    start_freq_hz: float
    stop_freq_hz: float
    step_hz: float = 100e3
    dwell_samples: int = 128 * 1024


# ── Device lifecycle ────────────────────────────────────────────────────────

@app.post("/device/open", summary="Open and configure the RTL-SDR device")
def open_device(req: OpenRequest):
    try:
        return sdr.open_device(req.sample_rate, req.center_freq, req.gain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/device/close", summary="Close the RTL-SDR device")
def close_device():
    return sdr.close_device()


@app.get("/device/status", summary="Get current device settings")
def device_status():
    return sdr.get_status()


# ── Tuning ──────────────────────────────────────────────────────────────────

@app.post("/tune/frequency", summary="Tune to a new center frequency")
def tune_frequency(req: FrequencyRequest):
    try:
        return sdr.set_frequency(req.freq_hz)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tune/gain", summary="Set receiver gain (dB or 'auto')")
def tune_gain(req: GainRequest):
    try:
        return sdr.set_gain(req.gain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tune/sample-rate", summary="Set sample rate in Hz")
def tune_sample_rate(req: SampleRateRequest):
    try:
        return sdr.set_sample_rate(req.rate_hz)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Spectrum ────────────────────────────────────────────────────────────────

@app.get("/spectrum", summary="Get power spectrum at current frequency")
def get_spectrum(num_samples: int = Query(default=256 * 1024, ge=1024)):
    try:
        return sdr.get_power_spectrum(num_samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/spectrum/scan", summary="Scan a frequency band")
def scan_band(req: ScanRequest):
    try:
        return sdr.scan_band(
            req.start_freq_hz, req.stop_freq_hz, req.step_hz, req.dwell_samples
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── FM demodulation ─────────────────────────────────────────────────────────

@app.get("/fm/demodulate", summary="Demodulate FM at current frequency")
def fm_demodulate(
    num_samples: int = Query(default=512 * 1024, ge=1024),
    include_audio: bool = Query(default=False, description="Include base64 WAV in response"),
):
    """
    Tunes the already-configured frequency, reads samples, and demodulates FM.
    Returns signal quality metrics and optionally a base64-encoded WAV file.
    """
    try:
        status = sdr.get_status()
        if not status["open"]:
            raise HTTPException(status_code=400, detail="Device not open")
        samples = sdr.read_samples(num_samples)
        return {
            "freq_hz": status["center_freq_hz"],
            **demodulate_fm_station(
                samples,
                sample_rate=status["sample_rate_hz"],
                include_audio=include_audio,
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fm/tune-and-demodulate", summary="Tune to frequency and demodulate FM")
def fm_tune_and_demodulate(
    req: FrequencyRequest,
    num_samples: int = Query(default=512 * 1024, ge=1024),
    include_audio: bool = Query(default=False),
):
    try:
        sdr.set_frequency(req.freq_hz)
        samples = sdr.read_samples(num_samples)
        status = sdr.get_status()
        return {
            "freq_hz": req.freq_hz,
            **demodulate_fm_station(
                samples,
                sample_rate=status["sample_rate_hz"],
                include_audio=include_audio,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
