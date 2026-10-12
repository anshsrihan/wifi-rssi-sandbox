#!/usr/bin/env python3
"""
Poll WiFi RSSI at a configurable interval and log to CSV.

Tier 1 only — read-only OS calls, no packet injection, no network disruption.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from sources import RssiReadError, describe_tier1_source, read_rssi_tier1

CSV_FIELDS = ["timestamp", "mac_or_label", "rssi_dbm", "source_tier", "source_method", "noise_dbm"]

_running = True


def _handle_stop(signum, frame) -> None:
    global _running
    _running = False


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_reading(path: Path, reading, timestamp: datetime) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writerow(
            {
                "timestamp": timestamp.isoformat(),
                "mac_or_label": reading.label,
                "rssi_dbm": reading.rssi_dbm,
                "source_tier": reading.source_tier,
                "source_method": reading.source_method,
                "noise_dbm": reading.noise_dbm if reading.noise_dbm is not None else "",
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log WiFi RSSI over time (Tier 1, read-only)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Polling interval in seconds (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/rssi_<timestamp>.csv)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Device label for CSV mac_or_label column",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Read a single sample and exit (useful for testing)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    interval = args.interval if args.interval is not None else config.get("polling_interval_sec", 1.0)
    label = args.label or config.get("device_label", "laptop")
    interface = config.get("interface")
    output_dir = Path(config.get("output_dir", "data"))

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"rssi_{stamp}.csv"
    else:
        output_path = args.output

    print(describe_tier1_source())
    print(f"Polling interval: {interval}s")
    print(f"Output file: {output_path.resolve()}")
    print("Press Ctrl+C to stop.\n")

    ensure_output_path(output_path)

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    start = time.monotonic()
    sample_count = 0
    consecutive_errors = 0

    while _running:
        now = datetime.now(timezone.utc)
        try:
            reading = read_rssi_tier1(label=label, interface=interface)
            append_reading(output_path, reading, now)
            sample_count += 1
            consecutive_errors = 0
            noise_str = f", noise={reading.noise_dbm} dBm" if reading.noise_dbm is not None else ""
            print(
                f"[{now.isoformat()}] {reading.label}: {reading.rssi_dbm:.0f} dBm "
                f"({reading.source_method}{noise_str})"
            )
        except RssiReadError as exc:
            consecutive_errors += 1
            print(f"ERROR: {exc}", file=sys.stderr)
            if consecutive_errors >= 5:
                print(
                    "\nFive consecutive read failures — stopping. "
                    "Check Wi-Fi connection and permissions.",
                    file=sys.stderr,
                )
                return 1

        if args.once:
            return 0 if consecutive_errors == 0 else 1

        if args.duration is not None and (time.monotonic() - start) >= args.duration:
            break

        time.sleep(interval)

    print(f"\nLogged {sample_count} samples to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
