# WiFi RSSI Signal Processing Sandbox

A local, experimental Python toolkit for studying WiFi signal strength (RSSI): logging, calibration, filtering, presence detection, and visualization. Runs entirely on your machine against your own home network — no cloud services, no telemetry.

## Safety constraints (non-negotiable)

This project is **read-only by design**:

1. **No deauth/disassociation packets** — nothing that disconnects or disrupts devices.
2. **No packet injection** — sniffing/capturing only; no crafted 802.11 frame transmission.
3. **No router configuration changes** — read-only status/API calls only (and only if you provide credentials).
4. **No tracking third-party devices** — Tier 1 reads your laptop's own connection; Tier 2 (disabled by default) would hash/anonymize non-owned MACs.
5. **No network-wide state changes** — no ARP spoofing, DHCP changes, firewall edits, etc.
6. **Elevated privileges require explicit approval** — monitor mode, `sudo wdutil`, etc. are gated and documented; nothing runs silently.

Audit check: `grep -rE 'sendp|send\(|deauth|disassoc' .` should find nothing unless you explicitly enable Tier 2 later.

## Requirements

- Python 3.11+
- macOS, Linux, or Windows
- Wi-Fi connected on the machine running the logger (Tier 1)

### Permissions

| Platform | Tier 1 RSSI | sudo needed? |
|----------|-------------|--------------|
| **macOS** | CoreWLAN via `swift` (fallback: `system_profiler`) | **No** |
| **Linux** | `iw` / `nmcli` | **No** for laptop RSSI; station dump from AP may vary |
| **Windows** | `netsh wlan show interfaces` | **No** |

Tier 2 (monitor mode, **off by default**) would require compatible hardware and often `sudo` — ask before enabling.

## Setup

```bash
cd wifi-rssi-sandbox
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project layout

```
wifi-rssi-sandbox/
  rssi_logger.py    # Poll RSSI → CSV
  calibrate.py      # Walk-test calibration + path-loss fit
  filters.py        # Moving avg, EMA, Kalman + comparison stats
  presence.py       # near / far / gone state machine
  dashboard.py      # Streamlit local dashboard
  sources.py        # Platform RSSI readers (Tier 1)
  config.yaml       # Thresholds, intervals, calibration paths
  data/             # Logged CSVs (gitignored)
```

## Quick start

### 1. Log RSSI

```bash
python rssi_logger.py --once          # single sample (smoke test)
python rssi_logger.py                 # poll until Ctrl+C
python rssi_logger.py --duration 60   # log for 60 seconds
```

Output: `data/rssi_YYYYMMDD_HHMMSS.csv` with columns  
`timestamp, mac_or_label, rssi_dbm, source_tier, source_method, noise_dbm`.

On **your Mac**, the primary reader is CoreWLAN via a built-in Swift one-liner (~300 ms per sample). The legacy `airport -I` tool is gone on recent macOS; `system_profiler` is used as fallback if Swift fails.

### 2. Calibrate distance model

Walk your phone (or laptop) to known distances from the AP:

```bash
python calibrate.py
```

Follow the prompts. Saves `calibration.json` and `data/calibration_fit.png`.

Model: `RSSI(d) = RSSI(d0) - 10 * n * log10(d / d0)`

### 3. Compare filters

```bash
python filters.py data/rssi_YYYYMMDD_HHMMSS.csv
```

Prints variance reduction and estimated lag for moving average, EMA, and Kalman.

### 4. Test presence classifier

```bash
python presence.py
```

Runs a synthetic near/far/gone sequence (no live Wi-Fi needed).

### 5. Dashboard

```bash
streamlit run dashboard.py
```

Opens on `http://localhost:8501`. Modes:

- **CSV replay** — visualize a past log with filters and presence timeline.
- **Live** — polls Tier 1 RSSI with auto-refresh (uses `@st.fragment`).

## Configuration

Edit `config.yaml` for polling interval, presence thresholds, filter parameters, and calibration distances.

Tier 2 flag (do not enable unless you explicitly want monitor mode):

```yaml
tier2:
  enable_monitor_mode: false
```

## macOS notes

- First Swift CoreWLAN call may take a few seconds while the compiler runs; subsequent reads are fast.
- SSID/BSSID may show as unavailable without Location Services permission — RSSI still works.
- `wdutil info` provides rich Wi-Fi stats but requires **sudo**; this project does not use it by default.

## Linux notes

- For per-client RSSI from an OpenWrt AP: `iw dev wlan0 station dump` (read-only).
- Laptop fallback: `nmcli device wifi` (signal percentage, approximated to dBm).

## What this is not

- Not a production indoor positioning system.
- Not a tool for scanning or tracking neighbors' devices.
- Not a router admin/automation tool.

## License

Personal learning project — use at your own discretion on networks you own or are permitted to test.
