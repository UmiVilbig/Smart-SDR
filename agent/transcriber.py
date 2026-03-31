"""
Whisper-based speech-to-text using faster-whisper.
Accepts float32 numpy audio arrays directly — no temp files.

Key settings for radio monitoring:
- condition_on_previous_text=False  → prevents hallucination feedback loops
- compression_ratio_threshold=1.8   → Whisper's built-in repetition filter
- VAD filter                        → skips silent chunks before feeding Whisper
- Post-transcription dedup          → drops windows that are near-identical to the last output
"""

import re
import difflib
import numpy as np
from scipy.signal import butter, sosfilt
from faster_whisper import WhisperModel

_model: WhisperModel | None = None
_model_name: str = ""
_prev_transcript: str = ""

# Phrases Whisper hallucinates on silence / static / music
_HALLUCINATION_PATTERNS = [
    r"^\s*$",
    r"^[\s♪♫\-\.]+$",
    r"thank you (for watching|for listening)",
    r"subtitles by",
    r"subscribe to",
    r"www\.",
    r"\.com",
    r"copyright",
    r"\[music\]",
    r"\[silence\]",
    r"\(music\)",
]
_HALLUCINATION_RE = re.compile(
    "|".join(_HALLUCINATION_PATTERNS), re.IGNORECASE
)


def load_model(model_name: str = "base.en") -> None:
    global _model, _model_name
    if _model is None or _model_name != model_name:
        print(f"[Transcriber] Loading Whisper model '{model_name}'...")
        _model = WhisperModel(model_name, device="cpu", compute_type="int8")
        _model_name = model_name
        print("[Transcriber] Model ready.")


def _bandpass_voice(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """80 Hz – 8 kHz bandpass: removes rumble and hiss, keeps full voice range."""
    nyq = sample_rate / 2.0
    sos = butter(4, [80.0 / nyq, min(8000.0 / nyq, 0.99)], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _similarity(a: str, b: str) -> float:
    """Return 0-1 similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _is_hallucination(text: str) -> bool:
    return bool(_HALLUCINATION_RE.search(text.strip()))


def transcribe(
    audio: np.ndarray,
    sample_rate: int = 16000,
    initial_prompt: str = "",
    dedup_threshold: float = 0.82,
) -> str:
    """
    Transcribe float32 audio at `sample_rate` Hz.

    Returns empty string if:
    - audio is silence
    - Whisper output looks like a hallucination
    - Output is too similar to the previous window (dedup)
    """
    global _prev_transcript

    if _model is None:
        load_model()

    # Pre-process
    audio = _bandpass_voice(audio, sample_rate)
    peak = np.max(np.abs(audio))
    if peak < 1e-5:
        return ""
    audio = audio / peak

    segments, info = _model.transcribe(
        audio,
        language="en",
        initial_prompt=initial_prompt or None,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 400,
            "threshold": 0.4,
            "speech_pad_ms": 150,
        },
        beam_size=5,
        # CRITICAL: False prevents the model feeding its own output back as a
        # prompt, which causes it to hallucinate / repeat the previous window.
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        # Tighter than default (2.4) — catches repetitive hallucinations early.
        compression_ratio_threshold=1.8,
        log_prob_threshold=-1.0,
    )

    text = " ".join(s.text.strip() for s in segments).strip()

    if not text:
        return ""

    if _is_hallucination(text):
        return ""

    # Dedup: if this window is almost identical to the last one, it's overlap noise
    if _similarity(text, _prev_transcript) > dedup_threshold:
        return ""

    _prev_transcript = text
    return text
