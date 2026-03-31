"""
Whisper-based speech-to-text using faster-whisper.
Accepts float32 numpy audio arrays directly — no temp files.
Includes voice-range bandpass filter to improve radio audio clarity.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from faster_whisper import WhisperModel

_model: WhisperModel | None = None
_model_name: str = ""
_last_transcript: str = ""   # fed back as context for overlapping windows


def load_model(model_name: str = "base.en") -> None:
    global _model, _model_name
    if _model is None or _model_name != model_name:
        print(f"[Transcriber] Loading Whisper model '{model_name}'...")
        _model = WhisperModel(model_name, device="cpu", compute_type="int8")
        _model_name = model_name
        print("[Transcriber] Model ready.")


def _bandpass_voice(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Bandpass 80 Hz – 8 kHz: removes sub-bass rumble and high-freq noise
    while keeping the full voice intelligibility range for Whisper.
    """
    nyq = sample_rate / 2.0
    low  = 80.0  / nyq
    high = min(8000.0 / nyq, 0.99)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def transcribe(
    audio: np.ndarray,
    sample_rate: int = 16000,
    initial_prompt: str = "",
) -> str:
    """
    Transcribe a float32 audio array at `sample_rate` Hz.
    Uses the previous result as context for better cross-chunk continuity.
    """
    global _last_transcript

    if _model is None:
        load_model()

    # Pre-process: filter then normalise
    audio = _bandpass_voice(audio, sample_rate)
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return ""   # pure silence / no signal
    audio = audio / peak

    # Prepend last transcript tail as context (helps Whisper at window edges)
    context = _last_transcript[-200:] if _last_transcript else initial_prompt

    segments, _ = _model.transcribe(
        audio,
        language="en",
        initial_prompt=context or initial_prompt or None,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 400,
            "threshold": 0.35,
            "speech_pad_ms": 200,
        },
        beam_size=5,
        condition_on_previous_text=True,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    if text:
        _last_transcript = text
    return text
