"""
RTL-SDR Radio Monitor
=====================
Continuously listens to a configured FM station, transcribes audio with Whisper,
detects trigger keywords (weather, traffic), summarizes with Claude, and posts
to Discord webhooks.

Usage:
    python -m agent.monitor                    # uses config.json
    python -m agent.monitor --config my.json   # custom config file
    python -m agent.monitor --station WTOP     # run only one station
    python -m agent.monitor --dry-run          # transcribe only, no Discord
"""

import argparse
import json
import os
import sys
import time
import signal
import threading
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ── DLL path must be registered before importing sdr_controller ─────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dll = os.path.join(_ROOT, "rtlsdr.dll")
if os.path.exists(_dll):
    os.add_dll_directory(_ROOT)

from api.sdr_controller import open_device, close_device, get_status, set_frequency, read_samples
from api.demodulation import fm_demodulate
from agent.transcriber import load_model, transcribe
from agent.discord_notifier import weather_embed, traffic_embed

# ── Globals ──────────────────────────────────────────────────────────────────
_running = True
_claude = anthropic.Anthropic()


def _handle_sigint(sig, frame):
    global _running
    print("\n[Monitor] Shutting down...")
    _running = False


signal.signal(signal.SIGINT, _handle_sigint)


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    config_path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    with open(config_path) as f:
        return json.load(f)


# ── Audio capture ─────────────────────────────────────────────────────────────

def capture_audio_chunk(sample_rate: float, chunk_seconds: float) -> np.ndarray:
    """Read IQ samples and FM-demodulate to 16 kHz float32 audio."""
    num_samples = int(chunk_seconds * sample_rate)
    # pyrtlsdr works best with multiples of 512; round up
    num_samples = ((num_samples + 511) // 512) * 512
    iq = read_samples(num_samples)
    audio = fm_demodulate(iq, sample_rate=sample_rate, audio_rate=16000)
    return audio


# ── Keyword detection ─────────────────────────────────────────────────────────

def contains_keywords(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


# ── Claude summarisation ──────────────────────────────────────────────────────

def summarize_transcript(transcript: str, prompt: str) -> str | None:
    """Ask Claude to summarise the relevant segment. Returns None if SKIP."""
    if not transcript.strip():
        return None

    msg = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\nTRANSCRIPT:\n{transcript}",
            }
        ],
    )
    result = msg.content[0].text.strip()
    if result.upper().startswith("SKIP"):
        return None
    return result


# ── Per-monitor state ─────────────────────────────────────────────────────────

class MonitorState:
    def __init__(self, cfg: dict, station_name: str):
        self.cfg = cfg
        self.station_name = station_name
        self.monitor_type = cfg["type"]
        self.keywords = cfg["trigger_keywords"]
        self.capture_extra = cfg["capture_extra_seconds"]
        self.cooldown = timedelta(minutes=cfg["cooldown_minutes"])
        self.webhook_url = cfg["discord_webhook_url"]
        self.mention = cfg.get("discord_mention", "")
        self.prompt = cfg["summarize_prompt"]

        self.last_sent: datetime | None = None
        self.capturing = False
        self.capture_start: datetime | None = None
        self.captured_text: list[str] = []
        self.last_keyword_time: datetime | None = None

    def on_cooldown(self) -> bool:
        if self.last_sent is None:
            return False
        return datetime.now() - self.last_sent < self.cooldown

    def process_transcript(self, text: str, dry_run: bool = False) -> None:
        if not text:
            return

        now = datetime.now()
        hits = contains_keywords(text, self.keywords)

        if hits:
            print(f"  [{self.monitor_type.upper()}] Keyword(s) detected: {hits}")
            self.last_keyword_time = now
            if not self.capturing:
                self.capturing = True
                self.capture_start = now
                self.captured_text = []
                print(f"  [{self.monitor_type.upper()}] Capture started — collecting for {self.capture_extra}s")

        if self.capturing:
            self.captured_text.append(text)
            elapsed = (now - self.capture_start).total_seconds()
            time_since_kw = (now - self.last_keyword_time).total_seconds() if self.last_keyword_time else 999

            # Stop capturing when: extra time elapsed OR no keyword seen for 20s
            if elapsed >= self.capture_extra or time_since_kw >= 20:
                full_transcript = " ".join(self.captured_text)
                print(f"  [{self.monitor_type.upper()}] Capture complete ({elapsed:.0f}s). Summarising...")
                self.capturing = False
                self.captured_text = []
                self._dispatch(full_transcript, dry_run)

    def _dispatch(self, transcript: str, dry_run: bool) -> None:
        if self.on_cooldown():
            remaining = (self.cooldown - (datetime.now() - self.last_sent)).seconds // 60
            print(f"  [{self.monitor_type.upper()}] On cooldown ({remaining}m remaining). Skipping.")
            return

        summary = summarize_transcript(transcript, self.prompt)
        if summary is None:
            print(f"  [{self.monitor_type.upper()}] Claude returned SKIP — no useful content.")
            return

        print(f"  [{self.monitor_type.upper()}] Summary:\n{summary}\n")

        if dry_run:
            print(f"  [{self.monitor_type.upper()}] [DRY RUN] Would post to Discord.")
            return

        send_fn = weather_embed if self.monitor_type == "weather" else traffic_embed
        ok = send_fn(
            summary=summary,
            station_name=self.station_name,
            webhook_url=self.webhook_url,
            mention=self.mention,
        )
        if ok:
            self.last_sent = datetime.now()
            print(f"  [{self.monitor_type.upper()}] Discord notification sent.")
        else:
            print(f"  [{self.monitor_type.upper()}] Discord send failed.")


# ── Station monitor loop ───────────────────────────────────────────────────────

def run_station(station_cfg: dict, sdr_cfg: dict, dry_run: bool = False) -> None:
    name = station_cfg["name"]
    freq = station_cfg["freq_hz"]
    sample_rate = sdr_cfg["sample_rate_hz"]
    chunk_sec = sdr_cfg["chunk_seconds"]

    enabled_monitors = [
        MonitorState(m, name)
        for m in station_cfg["monitors"]
        if m.get("enabled", True)
    ]

    if not enabled_monitors:
        print(f"[{name}] No enabled monitors — skipping.")
        return

    print(f"\n[{name}] Tuning to {freq/1e6:.1f} MHz | {len(enabled_monitors)} monitor(s) active")
    print(f"[{name}] Monitors: {[m.monitor_type for m in enabled_monitors]}")
    set_frequency(freq)

    chunk_num = 0
    while _running:
        try:
            audio = capture_audio_chunk(sample_rate, chunk_sec)
            text = transcribe(audio)
            chunk_num += 1

            ts = datetime.now().strftime("%H:%M:%S")
            if text:
                # Truncate long transcripts in the log
                preview = text[:120] + ("..." if len(text) > 120 else "")
                print(f"[{name}][{ts}] #{chunk_num}: {preview}")
            else:
                print(f"[{name}][{ts}] #{chunk_num}: (silence)")

            for monitor in enabled_monitors:
                monitor.process_transcript(text, dry_run=dry_run)

        except Exception as e:
            print(f"[{name}] Error in capture loop: {e}")
            time.sleep(2)


# ── Multi-station scheduler ───────────────────────────────────────────────────

def run_multi_station(stations: list[dict], sdr_cfg: dict, dry_run: bool = False) -> None:
    """
    For multiple stations, time-slice: spend `chunk_seconds * 4` on each station
    before rotating to the next. Single-station configs run continuously.
    """
    if len(stations) == 1:
        run_station(stations[0], sdr_cfg, dry_run)
        return

    # Build per-station monitor states (persist across rotations)
    station_states = {
        s["name"]: [
            MonitorState(m, s["name"])
            for m in s["monitors"]
            if m.get("enabled", True)
        ]
        for s in stations
    }

    chunk_sec = sdr_cfg["chunk_seconds"]
    sample_rate = sdr_cfg["sample_rate_hz"]
    chunks_per_slot = 4  # spend 4 chunks (~20s) on each station per rotation

    while _running:
        for station in stations:
            if not _running:
                break
            name = station["name"]
            freq = station["freq_hz"]
            monitors = station_states[name]

            print(f"\n[Scheduler] Switching to {name} ({freq/1e6:.1f} MHz)")
            set_frequency(freq)

            for _ in range(chunks_per_slot):
                if not _running:
                    break
                try:
                    audio = capture_audio_chunk(sample_rate, chunk_sec)
                    text = transcribe(audio)
                    ts = datetime.now().strftime("%H:%M:%S")
                    if text:
                        preview = text[:100] + ("..." if len(text) > 100 else "")
                        print(f"  [{name}][{ts}] {preview}")
                    for monitor in monitors:
                        monitor.process_transcript(text, dry_run=dry_run)
                except Exception as e:
                    print(f"  [{name}] Capture error: {e}")
                    time.sleep(1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RTL-SDR Radio Monitor")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--station", help="Run only this station name")
    parser.add_argument("--dry-run", action="store_true", help="Transcribe only, no Discord")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sdr_cfg = cfg["sdr"]

    # Load Whisper model upfront
    load_model(sdr_cfg.get("whisper_model", "base.en"))

    # Filter stations
    stations = [s for s in cfg["stations"] if s.get("enabled", True)]
    if args.station:
        stations = [s for s in stations if s["name"] == args.station]
    if not stations:
        print("No enabled stations found in config.")
        sys.exit(1)

    print(f"[Monitor] Opening SDR device...")
    open_device(
        sample_rate=sdr_cfg["sample_rate_hz"],
        center_freq=stations[0]["freq_hz"],
        gain=sdr_cfg.get("gain", "auto"),
    )
    print(f"[Monitor] Device open. Starting monitor loop. Ctrl+C to stop.\n")

    try:
        run_multi_station(stations, sdr_cfg, dry_run=args.dry_run)
    finally:
        close_device()
        print("[Monitor] Device closed.")


if __name__ == "__main__":
    main()
