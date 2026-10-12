#!/usr/bin/env python3
"""
Interactive RSSI calibration: walk-test at known distances, fit log-distance path loss.

Model: RSSI(d) = RSSI(d0) - 10 * n * log10(d / d0)

Read-only Tier 1 sampling — uses the same sources as rssi_logger.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import curve_fit

from sources import RssiReadError, describe_tier1_source, read_rssi_tier1


@dataclass
class CalibrationResult:
    rssi_d0: float
    path_loss_exponent: float
    d0_m: float
    distances_m: list[float]
    mean_rssi_by_distance: list[float]
    std_rssi_by_distance: list[float]
    fitted_at: str


def path_loss_model(d: np.ndarray, rssi_d0: float, n: float, d0: float) -> np.ndarray:
    return rssi_d0 - 10.0 * n * np.log10(d / d0)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def collect_samples_at_distance(
    distance_m: float,
    duration_sec: float,
    interval_sec: float,
    label: str,
    interface: str | None,
) -> list[float]:
    """Collect RSSI samples while user stands at a known distance."""
    print(f"\nStand at {distance_m:g} m from the AP/laptop reference point.")
    input("Press Enter when ready to start sampling...")
    print(f"Logging for {duration_sec:g} s (move as little as possible)...")

    samples: list[float] = []
    start = time.monotonic()
    while (time.monotonic() - start) < duration_sec:
        try:
            reading = read_rssi_tier1(label=label, interface=interface)
            samples.append(reading.rssi_dbm)
            print(f"  {reading.rssi_dbm:.0f} dBm", end="\r")
        except RssiReadError as exc:
            print(f"\n  Warning: {exc}", file=sys.stderr)
        time.sleep(interval_sec)

    print(f"\n  Collected {len(samples)} samples at {distance_m:g} m")
    return samples


def fit_calibration(
    distances_m: list[float],
    mean_rssi: list[float],
    d0_m: float,
) -> tuple[float, float]:
    """Fit path-loss exponent n and reference RSSI(d0)."""
    d = np.array(distances_m, dtype=float)
    rssi = np.array(mean_rssi, dtype=float)

    def model(d_vals, rssi_d0, n):
        return path_loss_model(d_vals, rssi_d0, n, d0_m)

    popt, _ = curve_fit(
        model,
        d,
        rssi,
        p0=[float(np.mean(rssi)), 2.0],
        bounds=([-120, 0.5], [-20, 6.0]),
        maxfev=10000,
    )
    return float(popt[0]), float(popt[1])


def plot_calibration(result: CalibrationResult, output_path: Path) -> None:
    d0 = result.d0_m
    d_meas = np.array(result.distances_m)
    d_curve = np.linspace(max(d0, 0.5), max(d_meas) * 1.1, 200)
    rssi_curve = path_loss_model(d_curve, result.rssi_d0, result.path_loss_exponent, d0)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        d_meas,
        result.mean_rssi_by_distance,
        yerr=result.std_rssi_by_distance,
        fmt="o",
        capsize=4,
        label="Measured mean ± std",
    )
    plt.plot(d_curve, rssi_curve, "-", label="Fitted log-distance model")
    plt.xlabel("Distance (m)")
    plt.ylabel("RSSI (dBm)")
    plt.title(
        f"Path-loss calibration: n={result.path_loss_exponent:.2f}, "
        f"RSSI(d0={d0}m)={result.rssi_d0:.1f} dBm"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path.resolve()}")


def estimate_distance(rssi_dbm: float, result: CalibrationResult) -> float:
    """Inverse of log-distance model: estimate distance from RSSI."""
    exponent = 10.0 * result.path_loss_exponent
    if exponent == 0:
        return result.d0_m
    ratio = 10.0 ** ((result.rssi_d0 - rssi_dbm) / exponent)
    return result.d0_m * ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive RSSI distance calibration.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=None,
        help="Distances in meters (overrides config)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("data/calibration_fit.png"),
        help="Output plot path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cal_cfg = config.get("calibration", {})

    distances = args.distances or cal_cfg.get("distances_m", [1, 2, 3, 5, 8, 10])
    d0_m = cal_cfg.get("d0_m", 1.0)
    duration = cal_cfg.get("sample_duration_sec", 10)
    interval = cal_cfg.get("sample_interval_sec", 0.5)
    output_file = Path(cal_cfg.get("output_file", "calibration.json"))
    label = config.get("device_label", "laptop")
    interface = config.get("interface")

    print("RSSI calibration — read-only Tier 1 sampling")
    print(describe_tier1_source())
    print(f"Reference distance d0 = {d0_m} m")
    print(f"Test distances: {distances}")

    mean_by_d: list[float] = []
    std_by_d: list[float] = []

    for distance_m in distances:
        samples = collect_samples_at_distance(
            distance_m, duration, interval, label, interface
        )
        if not samples:
            print(f"No samples collected at {distance_m} m — aborting.", file=sys.stderr)
            return 1
        mean_by_d.append(float(np.mean(samples)))
        std_by_d.append(float(np.std(samples)))

    rssi_d0, n = fit_calibration(distances, mean_by_d, d0_m)

    result = CalibrationResult(
        rssi_d0=rssi_d0,
        path_loss_exponent=n,
        d0_m=d0_m,
        distances_m=list(distances),
        mean_rssi_by_distance=mean_by_d,
        std_rssi_by_distance=std_by_d,
        fitted_at=datetime.now(timezone.utc).isoformat(),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)

    print("\n--- Calibration results ---")
    print(f"  RSSI(d0={d0_m} m) = {rssi_d0:.2f} dBm")
    print(f"  Path-loss exponent n = {n:.2f}")
    print(f"  Saved to {output_file.resolve()}")

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plot_calibration(result, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
