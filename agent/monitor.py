"""
RTL-SDR Radio Monitor
=====================
Architecture:
  CaptureThread  — continuously reads IQ from SDR, FM-demodulates to 16 kHz,
                   pushes audio chunks to a queue. Never stops or pauses.
  AudioPlayer    — optional thread that plays each chunk through speakers
                   in real time (--listen flag).
  WindowAssembler— assembles overlapping transcription windows from the queue.
  TranscribeLoop — calls Whisper on each window; dedup filter drops near-
                   identical consecutive outputs caused by window overlap.
  MonitorState   — keyword detection → LLM summarise → Discord webhook.

Usage:
    python -m agent.monitor                    # normal mode
    python -m agent.monitor --listen           # play audio through speakers
    python -m agent.monitor --dry-run          # no Discord, print only
    python -m agent.monitor --station WTOP     # single station
    python -m agent.monitor --config my.json   # custom config
"""

import argparse
import json
import os
import sys
import time
import signal
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── DLL path must be registered before any sdr import ────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(_ROOT, "rtlsdr.dll")):
    os.add_dll_directory(_ROOT)

from api.sdr_controller import open_device, close_device, set_frequency, read_samples
from api.demodulation import fm_demodulate
from agent.transcriber import load_model, transcribe
from agent.discord_notifier import send_monitor_embed
from agent.llm import summarize, check_ollama, list_models

# ── Shutdown flag ─────────────────────────────────────────────────────────────
_running = True


def _handle_sigint(sig, frame):
    global _running
    print("\n[Monitor] Stopping...")
    _running = False


signal.signal(signal.SIGINT, _handle_sigint)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    config_path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    with open(config_path) as f:
        return json.load(f)


# ── Audio playback ────────────────────────────────────────────────────────────

class AudioPlayer(threading.Thread):
    """
    Plays audio chunks through the default system output device.
    Receives chunks from a queue; drops old chunks if falling behind
    so playback stays in sync with capture.
    """

    AUDIO_RATE = 16000

    def __init__(self, max_queue: int = 8):
        super().__init__(daemon=True, name="AudioPlayer")
        self.play_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._sd = None

    def _load_sd(self):
        try:
            import sounddevice as sd
            self._sd = sd
            print(f"[Audio] Playback enabled → {sd.query_devices(kind='output')['name']}")
        except Exception as e:
            print(f"[Audio] sounddevice unavailable: {e}")

    def enqueue(self, audio: np.ndarray) -> None:
        """Non-blocking enqueue. Drops oldest if full to stay current."""
        if self.play_queue.full():
            try:
                self.play_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.play_queue.put_nowait(audio)
        except queue.Full:
            pass

    def stop(self):
        self._stop.set()

    def run(self):
        self._load_sd()
        if self._sd is None:
            return

        while not self._stop.is_set() and _running:
            try:
                audio = self.play_queue.get(timeout=1.0)
                # sounddevice expects float32 in [-1, 1]
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
                self._sd.play(audio, samplerate=self.AUDIO_RATE, blocking=True)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Audio] Playback error: {e}")


# ── Continuous capture thread ─────────────────────────────────────────────────

class CaptureThread(threading.Thread):
    """
    Reads IQ samples, FM-demodulates to 16 kHz audio, and pushes chunks to
    `audio_queue`. Also optionally copies chunks to `player` for live playback.
    """

    AUDIO_RATE = 16000

    def __init__(
        self,
        sample_rate: float,
        chunk_seconds: float,
        max_queue: int = 120,
        player: AudioPlayer | None = None,
    ):
        super().__init__(daemon=True, name="CaptureThread")
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.num_samples = ((int(chunk_seconds * sample_rate) + 511) // 512) * 512
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self.player = player
        self._stop_event = threading.Event()
        self.error_count = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        print(f"[Capture] Started — {self.num_samples} samples/chunk "
              f"({self.chunk_seconds}s @ {self.sample_rate/1e6:.3f} MHz)")
        while not self._stop_event.is_set() and _running:
            try:
                iq = read_samples(self.num_samples)
                audio = fm_demodulate(iq, sample_rate=self.sample_rate,
                                      audio_rate=self.AUDIO_RATE)

                # Feed speaker playback (separate queue, non-blocking)
                if self.player is not None:
                    self.player.enqueue(audio.copy())

                # Feed transcription queue; drop oldest if full
                if self.audio_queue.full():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.audio_queue.put_nowait(audio)
                self.error_count = 0

            except Exception as e:
                self.error_count += 1
                print(f"[Capture] Error ({self.error_count}): {e}")
                time.sleep(5 if self.error_count > 10 else 0.5)

        print("[Capture] Stopped.")


# ── Overlapping window assembler ──────────────────────────────────────────────

class WindowAssembler:
    """
    Pulls audio chunks from the capture queue and yields overlapping windows.
    window=15s, step=5s → 10s overlap — speech at boundaries is always caught.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        window_seconds: int = 15,
        step_seconds: int = 5,
        audio_rate: int = 16000,
    ):
        self.q = audio_queue
        self.window_samples = window_seconds * audio_rate
        self.step_samples = step_seconds * audio_rate
        self._buffer = np.array([], dtype=np.float32)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        while len(self._buffer) < self.window_samples and _running:
            try:
                chunk = self.q.get(timeout=1.0)
                self._buffer = np.concatenate([self._buffer, chunk])
            except queue.Empty:
                continue

        if not _running:
            raise StopIteration

        window = self._buffer[: self.window_samples].copy()
        self._buffer = self._buffer[self.step_samples :]
        return window


# ── Per-monitor state machine ─────────────────────────────────────────────────

class MonitorState:
    def __init__(self, cfg: dict, station_name: str, llm_cfg: dict):
        self.station_name = station_name
        self.monitor_type = cfg["type"]
        self.keywords = cfg["trigger_keywords"]
        self.capture_extra = cfg["capture_extra_seconds"]
        self.cooldown = timedelta(minutes=cfg["cooldown_minutes"])
        self.webhook_url = cfg["discord_webhook_url"]
        self.mention = cfg.get("discord_mention", "")
        self.prompt = cfg["summarize_prompt"]
        self.embed_title = cfg.get("embed_title", f"📻 {self.monitor_type.title()} Update")
        self.embed_color = cfg.get("embed_color", 0x5865F2)
        self.llm_cfg = llm_cfg

        self.last_sent: datetime | None = None
        self.capturing: bool = False
        self.capture_start: datetime | None = None
        self.last_keyword_time: datetime | None = None
        self.captured_segments: list[str] = []

    def on_cooldown(self) -> bool:
        return (
            self.last_sent is not None
            and datetime.now() - self.last_sent < self.cooldown
        )

    def process(self, text: str, dry_run: bool = False) -> None:
        if not text:
            return

        now = datetime.now()
        hits = [kw for kw in self.keywords if kw.lower() in text.lower()]

        if hits:
            print(f"  [{self.monitor_type.upper()}] keyword: {hits}")
            self.last_keyword_time = now
            if not self.capturing:
                self.capturing = True
                self.capture_start = now
                self.captured_segments = []
                print(f"  [{self.monitor_type.upper()}] Capture started "
                      f"(max {self.capture_extra}s)")

        if self.capturing:
            self.captured_segments.append(text)
            elapsed = (now - self.capture_start).total_seconds()
            since_kw = (now - self.last_keyword_time).total_seconds()

            if elapsed >= self.capture_extra or since_kw >= 20:
                full_text = " ".join(self.captured_segments)
                print(f"  [{self.monitor_type.upper()}] Capture ended "
                      f"({elapsed:.0f}s, {len(self.captured_segments)} segments) → summarising")
                self.capturing = False
                self.captured_segments = []
                self._dispatch(full_text, dry_run)

    def _dispatch(self, transcript: str, dry_run: bool) -> None:
        if self.on_cooldown():
            remaining = int(
                (self.cooldown - (datetime.now() - self.last_sent)).total_seconds() / 60
            )
            print(f"  [{self.monitor_type.upper()}] Cooldown {remaining}m remaining — skipped.")
            return

        llm = self.llm_cfg
        summary = summarize(
            transcript=transcript,
            prompt=self.prompt,
            model=llm.get("model", "llama3.2"),
            base_url=llm.get("base_url", "http://localhost:11434"),
            timeout=llm.get("timeout_seconds", 30),
            fallback_to_raw=llm.get("fallback_to_raw_transcript", True),
        )

        if summary is None:
            print(f"  [{self.monitor_type.upper()}] Nothing useful in transcript.")
            return

        print(f"  [{self.monitor_type.upper()}] Summary:\n{summary}\n")

        if dry_run:
            print(f"  [{self.monitor_type.upper()}] [DRY RUN] Discord skipped.")
            return

        if send_monitor_embed(
            summary=summary,
            station_name=self.station_name,
            webhook_url=self.webhook_url,
            title=self.embed_title,
            color=self.embed_color,
            mention=self.mention,
        ):
            self.last_sent = datetime.now()
            print(f"  [{self.monitor_type.upper()}] Discord sent ✓")


# ── Station runner ────────────────────────────────────────────────────────────

def run_station(
    station_cfg: dict,
    sdr_cfg: dict,
    llm_cfg: dict,
    dry_run: bool = False,
    player: AudioPlayer | None = None,
) -> None:
    name = station_cfg["name"]
    freq = station_cfg["freq_hz"]
    sample_rate = float(sdr_cfg["sample_rate_hz"])
    chunk_sec = float(sdr_cfg.get("capture_chunk_seconds", 2))
    window_sec = int(sdr_cfg.get("transcribe_window_seconds", 15))
    step_sec = int(sdr_cfg.get("transcribe_step_seconds", 5))
    initial_prompt = sdr_cfg.get("whisper_initial_prompt", "")

    monitors = [
        MonitorState(m, name, llm_cfg)
        for m in station_cfg["monitors"]
        if m.get("enabled", True)
    ]
    if not monitors:
        print(f"[{name}] No enabled monitors — skipping.")
        return

    print(f"\n[{name}] {freq/1e6:.1f} MHz | "
          f"window={window_sec}s step={step_sec}s overlap={window_sec - step_sec}s")
    print(f"[{name}] Monitors: {[m.monitor_type for m in monitors]}")
    set_frequency(freq)

    capture = CaptureThread(sample_rate, chunk_sec, player=player)
    capture.start()

    assembler = WindowAssembler(capture.audio_queue, window_sec, step_sec)
    window_num = 0
    for window_audio in assembler:
        if not _running:
            break
        window_num += 1
        ts = datetime.now().strftime("%H:%M:%S")
        text = transcribe(window_audio, initial_prompt=initial_prompt)
        if text:
            print(f"[{name}][{ts}] W{window_num}: {text[:120]}{'...' if len(text) > 120 else ''}")
        else:
            print(f"[{name}][{ts}] W{window_num}: (silence/filtered)")
        for m in monitors:
            m.process(text, dry_run=dry_run)

    capture.stop()


# ── Multi-station scheduler ───────────────────────────────────────────────────

def run_multi_station(
    stations: list[dict],
    sdr_cfg: dict,
    llm_cfg: dict,
    dry_run: bool = False,
    player: AudioPlayer | None = None,
) -> None:
    if len(stations) == 1:
        run_station(stations[0], sdr_cfg, llm_cfg, dry_run, player)
        return

    sample_rate = float(sdr_cfg["sample_rate_hz"])
    chunk_sec = float(sdr_cfg.get("capture_chunk_seconds", 2))
    window_sec = int(sdr_cfg.get("transcribe_window_seconds", 15))
    step_sec = int(sdr_cfg.get("transcribe_step_seconds", 5))
    initial_prompt = sdr_cfg.get("whisper_initial_prompt", "")
    windows_per_slot = max(1, 30 // step_sec)

    station_monitors = {
        s["name"]: [
            MonitorState(m, s["name"], llm_cfg)
            for m in s["monitors"] if m.get("enabled", True)
        ]
        for s in stations
    }

    while _running:
        for station in stations:
            if not _running:
                break
            name, freq = station["name"], station["freq_hz"]
            monitors = station_monitors[name]
            print(f"\n[Scheduler] → {name} ({freq/1e6:.1f} MHz)")
            set_frequency(freq)

            capture = CaptureThread(sample_rate, chunk_sec, player=player)
            capture.start()
            assembler = WindowAssembler(capture.audio_queue, window_sec, step_sec)

            for i, window_audio in enumerate(assembler):
                if not _running or i >= windows_per_slot:
                    break
                ts = datetime.now().strftime("%H:%M:%S")
                text = transcribe(window_audio, initial_prompt=initial_prompt)
                if text:
                    print(f"  [{name}][{ts}] {text[:100]}{'...' if len(text) > 100 else ''}")
                for m in monitors:
                    m.process(text, dry_run=dry_run)

            capture.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RTL-SDR Radio Monitor")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--station", help="Run only this station name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summaries only — do not post to Discord")
    parser.add_argument("--listen", action="store_true",
                        help="Play demodulated audio through speakers in real time")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sdr_cfg = cfg["sdr"]
    llm_cfg = cfg.get("llm", {})

    load_model(sdr_cfg.get("whisper_model", "base.en"))

    base_url = llm_cfg.get("base_url", "http://localhost:11434")
    if check_ollama(base_url):
        models = list_models(base_url)
        print(f"[LLM] Ollama online — models: {models}")
        want = llm_cfg.get("model", "llama3.2")
        if not any(want in m for m in models):
            print(f"[LLM] WARNING: '{want}' not found. Run: ollama pull {want}")
    else:
        print(f"[LLM] Ollama not reachable at {base_url}")
        if llm_cfg.get("fallback_to_raw_transcript", True):
            print("[LLM] Will post raw transcript excerpts as fallback.")

    stations = [s for s in cfg["stations"] if s.get("enabled", True)]
    if args.station:
        stations = [s for s in stations if s["name"] == args.station]
    if not stations:
        print("No enabled stations found.")
        sys.exit(1)

    # Start audio player if requested
    player: AudioPlayer | None = None
    if args.listen:
        player = AudioPlayer()
        player.start()

    print(f"\n[Monitor] Opening SDR...")
    open_device(
        sample_rate=sdr_cfg["sample_rate_hz"],
        center_freq=stations[0]["freq_hz"],
        gain=sdr_cfg.get("gain", "auto"),
    )
    print("[Monitor] Ready. Ctrl+C to stop.\n")

    try:
        run_multi_station(stations, sdr_cfg, llm_cfg,
                          dry_run=args.dry_run, player=player)
    finally:
        if player:
            player.stop()
        close_device()
        print("[Monitor] Device closed.")


if __name__ == "__main__":
    main()
