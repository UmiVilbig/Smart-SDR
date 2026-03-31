"""
RTL-SDR AI Agent
================
Uses Ollama (local LLM) with tool use to control the SDR,
demodulate signals, and post to Discord.

Usage:
    python -m agent.agent "Check 101.5 MHz FM signal quality"
    python -m agent.agent  # interactive loop
"""

import os
import sys
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("SDR_API_BASE", "http://localhost:8000")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ── Tool definitions (OpenAI-style for Ollama) ──────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_sdr",
            "description": "Open and initialise the RTL-SDR device. Call before any other SDR operation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "center_freq_hz": {"type": "number", "description": "Center frequency in Hz"},
                    "sample_rate_hz": {"type": "number", "description": "Sample rate in Hz"},
                    "gain": {"description": "Gain in dB or 'auto'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": "Return current SDR settings (frequency, gain, sample rate).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tune_frequency",
            "description": "Tune the SDR to a new center frequency.",
            "parameters": {
                "type": "object",
                "required": ["freq_hz"],
                "properties": {
                    "freq_hz": {"type": "number", "description": "Frequency in Hz"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spectrum",
            "description": "Get power spectrum at current frequency. Returns peak power and frequency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_samples": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "demodulate_fm",
            "description": "Demodulate FM audio at current frequency. Returns SNR and signal quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_samples": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tune_and_demodulate_fm",
            "description": "Tune to a frequency and demodulate FM in one step.",
            "parameters": {
                "type": "object",
                "required": ["freq_hz"],
                "properties": {
                    "freq_hz": {"type": "number"},
                    "num_samples": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_band",
            "description": "Scan a frequency range and return signal power at each step.",
            "parameters": {
                "type": "object",
                "required": ["start_freq_hz", "stop_freq_hz"],
                "properties": {
                    "start_freq_hz": {"type": "number"},
                    "stop_freq_hz": {"type": "number"},
                    "step_hz": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_discord",
            "description": "Post a message to a Discord webhook.",
            "parameters": {
                "type": "object",
                "required": ["message", "webhook_url"],
                "properties": {
                    "message": {"type": "string"},
                    "webhook_url": {"type": "string"},
                    "title": {"type": "string"},
                },
            },
        },
    },
]

# ── SDR REST helpers ─────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs) -> dict:
    with httpx.Client(timeout=60) as http:
        resp = getattr(http, method)(f"{API_BASE}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()


def execute_tool(name: str, inputs: dict) -> str:
    try:
        if name == "open_sdr":
            result = _api("post", "/device/open", json={
                "center_freq": inputs.get("center_freq_hz", 100.1e6),
                "sample_rate": inputs.get("sample_rate_hz", 1.024e6),
                "gain": inputs.get("gain", "auto"),
            })
        elif name == "get_device_status":
            result = _api("get", "/device/status")
        elif name == "tune_frequency":
            result = _api("post", "/tune/frequency", json={"freq_hz": inputs["freq_hz"]})
        elif name == "get_spectrum":
            result = _api("get", "/spectrum",
                          params={"num_samples": inputs.get("num_samples", 262144)})
        elif name == "demodulate_fm":
            result = _api("get", "/fm/demodulate",
                          params={"num_samples": inputs.get("num_samples", 524288)})
        elif name == "tune_and_demodulate_fm":
            freq = inputs["freq_hz"]
            result = _api("post", "/fm/tune-and-demodulate",
                          json={"freq_hz": freq},
                          params={"num_samples": inputs.get("num_samples", 524288)})
        elif name == "scan_band":
            result = _api("post", "/spectrum/scan", json=inputs)
        elif name == "send_discord":
            from agent.discord_notifier import send_discord
            ok = send_discord(
                webhook_url=inputs["webhook_url"],
                title=inputs.get("title", "SDR Report"),
                description=inputs["message"],
                station_name="SDR Agent",
            )
            result = {"sent": ok}
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Ollama agent loop ────────────────────────────────────────────────────────

SYSTEM = """You are an expert RF engineer assistant controlling an RTL-SDR radio receiver.

You have tools to open the device, tune to frequencies, capture spectra, and demodulate FM.
Always convert MHz to Hz when calling tools (e.g., 103.5 MHz = 103500000 Hz).
Be concise and technical."""


def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]
    print(f"\n[Agent] {user_message}\n")

    while True:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            text = msg.get("content", "")
            if text:
                print(f"[Agent] {text}")
            return text

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)

            print(f"[Tool] {name}({json.dumps(args)})")
            result = execute_tool(name, args)
            print(f"[Tool] → {result[:300]}")

            messages.append({
                "role": "tool",
                "content": result,
            })


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_agent(" ".join(sys.argv[1:]))
    else:
        print(f"RTL-SDR Agent (Ollama/{OLLAMA_MODEL}) — type 'quit' to exit\n")
        while True:
            try:
                msg = input("You: ").strip()
                if msg.lower() in ("quit", "exit", "q"):
                    break
                if msg:
                    run_agent(msg)
            except KeyboardInterrupt:
                break
