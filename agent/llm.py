"""
Local LLM interface via Ollama.
Falls back to returning a trimmed raw transcript if Ollama is unreachable.
"""

import httpx
import json


def summarize(
    transcript: str,
    prompt: str,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    timeout: float = 30.0,
    fallback_to_raw: bool = True,
) -> str | None:
    """
    Ask the local LLM to summarise a transcript segment.
    Returns None if the model says there's nothing relevant (SKIP).
    Returns a trimmed raw excerpt as fallback if Ollama is unreachable.
    """
    if not transcript.strip():
        return None

    full_prompt = f"{prompt}\n\nTRANSCRIPT:\n{transcript}"

    try:
        resp = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 400},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()["message"]["content"].strip()

        if result.upper().startswith("SKIP"):
            return None
        return result

    except httpx.ConnectError:
        if fallback_to_raw:
            print("[LLM] Ollama not reachable — falling back to raw transcript excerpt.")
            return _raw_excerpt(transcript)
        print("[LLM] Ollama not reachable and fallback disabled.")
        return None
    except Exception as e:
        print(f"[LLM] Error calling Ollama: {e}")
        if fallback_to_raw:
            return _raw_excerpt(transcript)
        return None


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Return True if Ollama is running and reachable."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def list_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Return list of locally available Ollama model names."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def _raw_excerpt(transcript: str, max_chars: int = 800) -> str:
    """Return a trimmed version of the transcript for fallback Discord posts."""
    trimmed = transcript.strip()[:max_chars]
    if len(transcript) > max_chars:
        trimmed += "..."
    return f"*(raw transcript — Ollama offline)*\n{trimmed}"
