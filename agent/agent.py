"""
RTL-SDR AI Agent
================
Uses Claude with tool use to control the SDR, demodulate signals,
analyse content, and send SMS reports.

Usage:
    python -m agent.agent "Check 101.5 MHz FM and text me the weather"
    python -m agent.agent  # interactive loop
"""

import os
import sys
import json
import httpx
import anthropic
from dotenv import load_dotenv
from agent.sms import send_sms

load_dotenv()

API_BASE = os.getenv("SDR_API_BASE", "http://localhost:8000")
SMS_TO = os.getenv("SMS_TO_NUMBER", "")  # your phone number e.g. +1XXXXXXXXXX

client = anthropic.Anthropic()

# ── Tool definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "open_sdr",
        "description": (
            "Open and initialise the RTL-SDR device. "
            "Must be called before any other SDR operation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "center_freq_hz": {
                    "type": "number",
                    "description": "Initial center frequency in Hz (e.g. 100.1e6 for 100.1 MHz)",
                },
                "sample_rate_hz": {
                    "type": "number",
                    "description": "Sample rate in Hz, default 2048000",
                },
                "gain": {
                    "description": "Gain in dB or 'auto'",
                    "oneOf": [{"type": "number"}, {"type": "string"}],
                },
            },
        },
    },
    {
        "name": "get_device_status",
        "description": "Return current SDR device settings (frequency, gain, sample rate).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "tune_frequency",
        "description": "Tune the SDR to a new center frequency.",
        "input_schema": {
            "type": "object",
            "required": ["freq_hz"],
            "properties": {
                "freq_hz": {
                    "type": "number",
                    "description": "Frequency in Hz (e.g. 101.5e6 for 101.5 MHz FM)",
                }
            },
        },
    },
    {
        "name": "get_spectrum",
        "description": (
            "Get the power spectrum at the current frequency. "
            "Returns peak frequency and power in dB."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "num_samples": {"type": "integer", "description": "Samples to collect, default 262144"}
            },
        },
    },
    {
        "name": "demodulate_fm",
        "description": (
            "Demodulate FM audio at the current frequency. "
            "Returns signal quality metrics (SNR, power) and duration. "
            "Set include_audio=true to get a base64 WAV for playback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "num_samples": {"type": "integer", "description": "IQ samples to collect, default 524288"},
                "include_audio": {"type": "boolean", "description": "Include base64 WAV in result"},
            },
        },
    },
    {
        "name": "tune_and_demodulate_fm",
        "description": "Tune to a frequency and immediately demodulate FM audio.",
        "input_schema": {
            "type": "object",
            "required": ["freq_hz"],
            "properties": {
                "freq_hz": {"type": "number"},
                "num_samples": {"type": "integer"},
                "include_audio": {"type": "boolean"},
            },
        },
    },
    {
        "name": "scan_band",
        "description": (
            "Scan a frequency range and return signal power at each step. "
            "Useful for finding active FM stations."
        ),
        "input_schema": {
            "type": "object",
            "required": ["start_freq_hz", "stop_freq_hz"],
            "properties": {
                "start_freq_hz": {"type": "number"},
                "stop_freq_hz": {"type": "number"},
                "step_hz": {"type": "number", "description": "Step size in Hz, default 100000"},
            },
        },
    },
    {
        "name": "send_sms",
        "description": "Send an SMS message to the configured phone number.",
        "input_schema": {
            "type": "object",
            "required": ["message"],
            "properties": {
                "message": {"type": "string", "description": "The SMS body to send"},
                "to": {
                    "type": "string",
                    "description": "Override destination number (E.164 format). Uses SMS_TO_NUMBER env var if omitted.",
                },
            },
        },
    },
]

# ── Tool execution ──────────────────────────────────────────────────────────

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
                "sample_rate": inputs.get("sample_rate_hz", 2.048e6),
                "gain": inputs.get("gain", "auto"),
            })
        elif name == "get_device_status":
            result = _api("get", "/device/status")
        elif name == "tune_frequency":
            result = _api("post", "/tune/frequency", json={"freq_hz": inputs["freq_hz"]})
        elif name == "get_spectrum":
            params = {}
            if "num_samples" in inputs:
                params["num_samples"] = inputs["num_samples"]
            result = _api("get", "/spectrum", params=params)
        elif name == "demodulate_fm":
            params = {k: v for k, v in inputs.items() if k != "include_audio"}
            params["include_audio"] = inputs.get("include_audio", False)
            result = _api("get", "/fm/demodulate", params=params)
        elif name == "tune_and_demodulate_fm":
            freq = inputs.pop("freq_hz")
            params = {k: v for k, v in inputs.items()}
            result = _api("post", "/fm/tune-and-demodulate",
                          json={"freq_hz": freq}, params=params)
        elif name == "scan_band":
            result = _api("post", "/spectrum/scan", json=inputs)
        elif name == "send_sms":
            to = inputs.get("to") or SMS_TO
            if not to:
                return json.dumps({"error": "No destination number. Set SMS_TO_NUMBER in .env"})
            result = send_sms(to=to, body=inputs["message"])
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Agent loop ──────────────────────────────────────────────────────────────

SYSTEM = """You are an expert RF engineer assistant controlling an RTL-SDR radio receiver.

You have tools to:
- Open and configure the SDR device
- Tune to any frequency
- Capture power spectra and scan bands
- Demodulate FM radio stations
- Send SMS messages

When a user asks you to monitor a station:
1. Open the device if not already open
2. Tune to the requested frequency
3. Demodulate FM to get signal quality
4. Describe what you observe about the signal (quality, strength, whether audio is clean)
5. If the user wants weather or traffic: explain that real-time transcription requires additional
   speech-to-text integration (Whisper), but describe the signal quality and what you can infer
6. Send an SMS summary if requested

Always convert MHz to Hz when calling tools (e.g., 101.5 MHz = 101500000 Hz).
Be concise and technical but friendly.
"""


def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    print(f"\n[Agent] User: {user_message}\n")

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any text blocks
        text_output = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_output += block.text
            elif block.type == "tool_use":
                tool_calls.append(block)

        if text_output:
            print(f"[Agent] {text_output}")

        if response.stop_reason == "end_turn" or not tool_calls:
            return text_output

        # Execute all tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            print(f"[Tool] {tc.name}({json.dumps(tc.input, indent=2)})")
            result = execute_tool(tc.name, tc.input)
            parsed = json.loads(result)
            print(f"[Tool] → {json.dumps(parsed, indent=2)[:500]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


# ── CLI entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_agent(" ".join(sys.argv[1:]))
    else:
        print("RTL-SDR AI Agent — type 'quit' to exit\n")
        while True:
            try:
                msg = input("You: ").strip()
                if msg.lower() in ("quit", "exit", "q"):
                    break
                if msg:
                    run_agent(msg)
            except KeyboardInterrupt:
                break
