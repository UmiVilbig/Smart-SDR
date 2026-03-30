# RTL-SDR REST API + AI Agent

## Setup

### 1. Install RTL-SDR Windows drivers
1. Download [Zadig](https://zadig.akeo.ie/) and run it
2. Plug in your RTL-SDR dongle
3. In Zadig: **Options → List All Devices**, select your RTL-SDR, install **WinUSB** driver
4. Download `rtlsdr.dll` from the [RTL-SDR releases](https://github.com/osmocom/rtl-sdr/releases)
   and place it next to `api/main.py` **or** add its folder to your PATH

### 2. Configure environment
```
cp .env.example .env
# Edit .env with your API keys
```

### 3. Install Python dependencies
```
pip install -r requirements.txt
```

### 4. Start the REST API
```
uvicorn api.main:app --reload --port 8000
```
Browse the interactive docs at http://localhost:8000/docs

### 5. Run the AI Agent
```
# Single command
python -m agent.agent "Tune to 101.5 MHz FM and text me the signal quality"

# Interactive mode
python -m agent.agent
```

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/device/open` | Open RTL-SDR device |
| POST | `/device/close` | Close device |
| GET  | `/device/status` | Current settings |
| POST | `/tune/frequency` | Tune to frequency (Hz) |
| POST | `/tune/gain` | Set gain (dB or "auto") |
| POST | `/tune/sample-rate` | Set sample rate |
| GET  | `/spectrum` | Power spectrum at current freq |
| POST | `/spectrum/scan` | Scan a frequency band |
| GET  | `/fm/demodulate` | FM demodulate current freq |
| POST | `/fm/tune-and-demodulate` | Tune + demodulate in one call |

## Example agent prompts
- `"Open the SDR and scan FM band 88-108 MHz, find the strongest station"`
- `"Tune to 162.4 MHz (NOAA weather) and check signal strength, then text me"`
- `"Check 101.5 MHz FM signal quality and send me an SMS summary"`
