"""
RSSI noise filters: moving average, exponential moving average, and 1D Kalman.

All functions accept pandas Series or numpy arrays and return filtered arrays
of the same length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.Series, list[float]]


def _to_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def moving_average(values: ArrayLike, window: int = 5) -> np.ndarray:
    """Simple centered moving average with edge padding via partial windows."""
    arr = _to_array(values)
    if window <= 1:
        return arr.copy()
    if len(arr) == 0:
        return arr.copy()

    series = pd.Series(arr)
    return series.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def exponential_moving_average(values: ArrayLike, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average; higher alpha = less smoothing, less lag."""
    arr = _to_array(values)
    if len(arr) == 0:
        return arr.copy()

    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


class KalmanFilter1D:
    """
    Basic 1D Kalman filter for scalar RSSI measurements.

    State: signal level (dBm). Constant-velocity model is overkill for RSSI;
    we use a random-walk model with process noise.
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 4.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = 0.0
        self.error_covariance = 1.0
        self._initialized = False

    def update(self, measurement: float) -> float:
        if not self._initialized:
            self.estimate = measurement
            self._initialized = True
            return self.estimate

        # Predict
        predicted_covariance = self.error_covariance + self.process_noise

        # Update
        kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_noise)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_covariance = (1.0 - kalman_gain) * predicted_covariance
        return self.estimate


def kalman_filter(values: ArrayLike, process_noise: float = 1.0, measurement_noise: float = 4.0) -> np.ndarray:
    """Apply a 1D Kalman filter across a sequence of RSSI values."""
    arr = _to_array(values)
    kf = KalmanFilter1D(process_noise=process_noise, measurement_noise=measurement_noise)
    return np.array([kf.update(v) for v in arr])


def apply_all_filters(
    values: ArrayLike,
    ma_window: int = 5,
    ema_alpha: float = 0.3,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 4.0,
) -> dict[str, np.ndarray]:
    """Run all three filters and return a dict of filtered series."""
    arr = _to_array(values)
    return {
        "raw": arr,
        "moving_average": moving_average(arr, window=ma_window),
        "ema": exponential_moving_average(arr, alpha=ema_alpha),
        "kalman": kalman_filter(
            arr,
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
        ),
    }


@dataclass
class FilterComparisonStats:
    name: str
    variance: float
    variance_reduction_pct: float
    mean_abs_lag: float


def _estimate_lag(raw: np.ndarray, filtered: np.ndarray) -> float:
    """
    Estimate filter lag by finding the shift that maximizes correlation.

    Returns mean absolute lag in sample units (positive = filter lags behind).
    """
    if len(raw) < 3:
        return 0.0

    raw_centered = raw - np.mean(raw)
    filt_centered = filtered - np.mean(filtered)
    max_shift = min(20, len(raw) // 4)
    best_shift = 0
    best_corr = -1.0

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a = raw_centered[shift:]
            b = filt_centered[: len(a)]
        else:
            b = filt_centered[-shift:]
            a = raw_centered[: len(b)]
        if len(a) < 2:
            continue
        denom = np.std(a) * np.std(b)
        if denom == 0:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return float(abs(best_shift))


def compare_filters(
    values: ArrayLike,
    ma_window: int = 5,
    ema_alpha: float = 0.3,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 4.0,
) -> list[FilterComparisonStats]:
    """Compute variance reduction and estimated lag for each filter."""
    filtered = apply_all_filters(
        values,
        ma_window=ma_window,
        ema_alpha=ema_alpha,
        kalman_process_noise=kalman_process_noise,
        kalman_measurement_noise=kalman_measurement_noise,
    )
    raw = filtered["raw"]
    raw_var = float(np.var(raw))
    stats: list[FilterComparisonStats] = []

    for name in ("moving_average", "ema", "kalman"):
        series = filtered[name]
        var = float(np.var(series))
        reduction = 0.0 if raw_var == 0 else (1.0 - var / raw_var) * 100.0
        stats.append(
            FilterComparisonStats(
                name=name,
                variance=var,
                variance_reduction_pct=reduction,
                mean_abs_lag=_estimate_lag(raw, series),
            )
        )
    return stats


def load_rssi_from_csv(path: Path, column: str = "rssi_dbm") -> pd.Series:
    """Load RSSI values from a logger CSV file."""
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {path}")
    return df[column].astype(float)


def compare_filters_from_csv(csv_path: Path, config: dict | None = None) -> pd.DataFrame:
    """Load a CSV, apply filters, and return a comparison DataFrame."""
    config = config or {}
    filter_cfg = config.get("filters", {})
    values = load_rssi_from_csv(csv_path)
    stats = compare_filters(
        values,
        ma_window=filter_cfg.get("moving_average_window", 5),
        ema_alpha=filter_cfg.get("ema_alpha", 0.3),
        kalman_process_noise=filter_cfg.get("kalman_process_noise", 1.0),
        kalman_measurement_noise=filter_cfg.get("kalman_measurement_noise", 4.0),
    )
    return pd.DataFrame(
        [
            {
                "filter": s.name,
                "variance": s.variance,
                "variance_reduction_pct": s.variance_reduction_pct,
                "estimated_lag_samples": s.mean_abs_lag,
            }
            for s in stats
        ]
    )


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Compare RSSI filters on a logged CSV.")
    parser.add_argument("csv", type=Path, help="Path to rssi_logger CSV output")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()

    cfg = {}
    if args.config.exists():
        with args.config.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}

    df = compare_filters_from_csv(args.csv, cfg)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
