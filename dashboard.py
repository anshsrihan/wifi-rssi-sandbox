#!/usr/bin/env python3
"""
Local-only RSSI dashboard (Streamlit, localhost).

Shows raw vs filtered RSSI, presence state, and estimated distance if calibrated.
No external network calls, no telemetry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from calibrate import CalibrationResult, estimate_distance, path_loss_model
from filters import apply_all_filters
from presence import PresenceClassifier, PresenceConfig
from sources import RssiReadError, describe_tier1_source, read_rssi_tier1

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def load_config() -> dict:
    if DEFAULT_CONFIG.exists():
        with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def load_calibration(path: Path) -> CalibrationResult | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return CalibrationResult(**data)


def load_csv_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


def main() -> None:
    st.set_page_config(page_title="WiFi RSSI Sandbox", layout="wide")
    st.title("WiFi RSSI Sandbox")
    st.caption("Local-only dashboard — no cloud, no telemetry")

    config = load_config()
    filter_cfg = config.get("filters", {})
    presence_cfg = PresenceConfig(**config.get("presence", {}))
    cal_path = Path(config.get("calibration", {}).get("output_file", "calibration.json"))
    calibration = load_calibration(PROJECT_DIR / cal_path)

    st.sidebar.header("Mode")
    mode = st.sidebar.radio("Data source", ["Live", "CSV replay"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Tier 1 source:**  \n{describe_tier1_source()}")

    if calibration:
        st.sidebar.success(
            f"Calibration loaded: n={calibration.path_loss_exponent:.2f}, "
            f"RSSI(d0)={calibration.rssi_d0:.1f} dBm"
        )
    else:
        st.sidebar.warning("No calibration.json — distance estimate disabled.")

    if mode == "CSV replay":
        data_dir = PROJECT_DIR / config.get("output_dir", "data")
        csv_files = sorted(data_dir.glob("rssi_*.csv"), reverse=True)
        if not csv_files:
            st.error(f"No CSV logs found in {data_dir}. Run rssi_logger.py first.")
            return

        selected = st.sidebar.selectbox("CSV file", csv_files, format_func=lambda p: p.name)
        df = load_csv_log(selected)
        st.subheader(f"Replay: {selected.name}")

        filtered = apply_all_filters(
            df["rssi_dbm"].values,
            ma_window=filter_cfg.get("moving_average_window", 5),
            ema_alpha=filter_cfg.get("ema_alpha", 0.3),
            kalman_process_noise=filter_cfg.get("kalman_process_noise", 1.0),
            kalman_measurement_noise=filter_cfg.get("kalman_measurement_noise", 4.0),
        )

        clf = PresenceClassifier(presence_cfg)
        timestamps = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().tolist()
        states = []
        for ts, rssi in zip(timestamps, filtered["kalman"]):
            states.append(clf.update(float(rssi), float(ts)).value)

        plot_df = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "raw": filtered["raw"],
                "moving_average": filtered["moving_average"],
                "ema": filtered["ema"],
                "kalman": filtered["kalman"],
                "presence": states,
            }
        )

        if calibration:
            plot_df["distance_m"] = plot_df["kalman"].apply(
                lambda r: estimate_distance(r, calibration)
            )

        _render_charts(plot_df, calibration)
        st.dataframe(plot_df.tail(20), use_container_width=True)
        return

    interval = st.sidebar.slider("Poll interval (s)", 0.5, 5.0, float(config.get("polling_interval_sec", 1.0)))
    max_points = st.sidebar.slider("History length", 30, 500, 120)
    label = config.get("device_label", "laptop")
    interface = config.get("interface")

    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.classifier = PresenceClassifier(presence_cfg)
        st.session_state.live_started_at = time.time()

    if st.sidebar.button("Clear live history"):
        st.session_state.history = []
        st.session_state.classifier.reset()
        st.session_state.live_started_at = time.time()

    @st.fragment(run_every=interval)
    def live_poll() -> None:
        try:
            reading = read_rssi_tier1(label=label, interface=interface)
            st.session_state.history.append(
                {
                    "timestamp": pd.Timestamp.now(tz="UTC"),
                    "raw": reading.rssi_dbm,
                    "label": reading.label,
                    "method": reading.source_method,
                }
            )
            if len(st.session_state.history) > max_points:
                st.session_state.history = st.session_state.history[-max_points:]
        except RssiReadError as exc:
            st.error(str(exc))
            return

        hist = st.session_state.history
        if not hist:
            return

        raw = [h["raw"] for h in hist]
        filtered = apply_all_filters(
            raw,
            ma_window=filter_cfg.get("moving_average_window", 5),
            ema_alpha=filter_cfg.get("ema_alpha", 0.3),
            kalman_process_noise=filter_cfg.get("kalman_process_noise", 1.0),
            kalman_measurement_noise=filter_cfg.get("kalman_measurement_noise", 4.0),
        )

        elapsed = [h["timestamp"].timestamp() - st.session_state.live_started_at for h in hist]
        current_kalman = float(filtered["kalman"][-1])
        state = st.session_state.classifier.update(current_kalman, elapsed[-1])

        plot_df = pd.DataFrame(
            {
                "timestamp": [h["timestamp"] for h in hist],
                "raw": filtered["raw"],
                "moving_average": filtered["moving_average"],
                "ema": filtered["ema"],
                "kalman": filtered["kalman"],
            }
        )

        dist_str = ""
        if calibration:
            dist = estimate_distance(current_kalman, calibration)
            dist_str = f" | est. distance: {dist:.1f} m"

        st.info(
            f"**{state.value.upper()}** — "
            f"raw: {raw[-1]:.0f} dBm, kalman: {current_kalman:.0f} dBm{dist_str} "
            f"({hist[-1]['method']})"
        )
        _render_charts(plot_df, calibration, live_state=state.value)

    live_poll()


def _render_charts(
    plot_df: pd.DataFrame,
    calibration: CalibrationResult | None,
    live_state: str | None = None,
) -> None:
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = plot_df["timestamp"] if "timestamp" in plot_df.columns else range(len(plot_df))
        ax.plot(x, plot_df["raw"], alpha=0.35, label="raw", linewidth=1)
        ax.plot(x, plot_df["moving_average"], label="moving avg", linewidth=1.5)
        ax.plot(x, plot_df["ema"], label="EMA", linewidth=1.5)
        ax.plot(x, plot_df["kalman"], label="Kalman", linewidth=2)
        ax.set_ylabel("RSSI (dBm)")
        ax.set_xlabel("Time")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        title = "RSSI over time (raw vs filtered)"
        if live_state:
            title += f" — presence: {live_state}"
        ax.set_title(title)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        if calibration and "distance_m" in plot_df.columns:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.plot(plot_df["distance_m"], plot_df["kalman"], "o-", markersize=3)
            ax2.set_xlabel("Est. distance (m)")
            ax2.set_ylabel("RSSI (dBm)")
            ax2.set_title("RSSI vs estimated distance")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)
        elif calibration:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            d = np.linspace(calibration.d0_m, max(calibration.distances_m) * 1.1, 100)
            rssi = path_loss_model(
                d, calibration.rssi_d0, calibration.path_loss_exponent, calibration.d0_m
            )
            ax2.plot(d, rssi, label="calibration curve")
            ax2.scatter(
                calibration.distances_m,
                calibration.mean_rssi_by_distance,
                color="red",
                zorder=5,
                label="measured",
            )
            ax2.set_xlabel("Distance (m)")
            ax2.set_ylabel("RSSI (dBm)")
            ax2.legend()
            ax2.set_title("Calibration curve")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Run calibrate.py to enable distance plots.")

        if "presence" in plot_df.columns:
            st.markdown("**Presence timeline (last samples)**")
            st.code("\n".join(plot_df["presence"].tail(15).tolist()))


if __name__ == "__main__":
    main()
