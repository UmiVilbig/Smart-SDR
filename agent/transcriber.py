"""
Whisper-based speech-to-text using faster-whisper.
Converts float32 audio arrays directly — no temp files needed.
"""

import numpy as np
from faster_whisper import WhisperModel

_model: WhisperModel | None = None
_model_name: str = ""


def load_model(model_name: str = "base.en") -> None:
    global _model, _model_name
    if _model is None or _model_name != model_name:
        print(f"[Transcriber] Loading Whisper model '{model_name}'...")
        _model = WhisperModel(model_name, device="cpu", compute_type="int8")
        _model_name = model_name
        print("[Transcriber] Model ready.")


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Transcribe a float32 audio array.
    faster-whisper expects float32 at 16 kHz mono.
    """
    if _model is None:
        load_model()

    # Ensure float32, normalised to [-1, 1]
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    segments, _ = _model.transcribe(
        audio,
        language="en",
        vad_filter=True,          # skip silent chunks
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
