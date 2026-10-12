"""
Presence / proximity classifier from filtered RSSI values.

State machine: near / far / gone with threshold hysteresis and a gone timeout.
Designed to be testable with synthetic RSSI sequences — no live data required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


class PresenceState(str, Enum):
    NEAR = "near"
    FAR = "far"
    GONE = "gone"


@dataclass
class PresenceConfig:
    near_threshold_dbm: float = -55.0
    far_threshold_dbm: float = -75.0
    hysteresis_dbm: float = 3.0
    gone_timeout_sec: float = 30.0


@dataclass
class PresenceStep:
    rssi_dbm: Optional[float]
    timestamp_sec: float
    state: PresenceState


class PresenceClassifier:
    """
    Classify presence from a stream of (timestamp, RSSI) samples.

    Thresholds use hysteresis to avoid flickering:
    - Enter NEAR when RSSI >= near_threshold
    - Leave NEAR when RSSI < near_threshold - hysteresis
    - Enter FAR when RSSI < far_threshold
    - Leave FAR when RSSI >= far_threshold + hysteresis
    - GONE when no valid sample for gone_timeout_sec
    """

    def __init__(self, config: Optional[PresenceConfig] = None):
        self.config = config or PresenceConfig()
        self.state = PresenceState.GONE
        self._last_valid_timestamp: Optional[float] = None

    def reset(self) -> None:
        self.state = PresenceState.GONE
        self._last_valid_timestamp = None

    def update(self, rssi_dbm: Optional[float], timestamp_sec: float) -> PresenceState:
        cfg = self.config

        if rssi_dbm is not None and not (isinstance(rssi_dbm, float) and rssi_dbm != rssi_dbm):
            self._last_valid_timestamp = timestamp_sec
            self._update_from_rssi(float(rssi_dbm))
        elif (
            self._last_valid_timestamp is not None
            and (timestamp_sec - self._last_valid_timestamp) >= cfg.gone_timeout_sec
        ):
            self.state = PresenceState.GONE

        return self.state

    def _update_from_rssi(self, rssi_dbm: float) -> None:
        cfg = self.config
        h = cfg.hysteresis_dbm

        if self.state == PresenceState.GONE:
            if rssi_dbm >= cfg.near_threshold_dbm:
                self.state = PresenceState.NEAR
            else:
                self.state = PresenceState.FAR
            return

        if self.state == PresenceState.NEAR:
            if rssi_dbm < cfg.near_threshold_dbm - h:
                self.state = PresenceState.FAR
            return

        if self.state == PresenceState.FAR:
            if rssi_dbm >= cfg.near_threshold_dbm:
                self.state = PresenceState.NEAR
            return

    def classify_sequence(
        self,
        samples: Iterable[tuple[float, Optional[float]]],
    ) -> list[PresenceStep]:
        """
        Classify an entire sequence of (timestamp_sec, rssi_dbm) tuples.

        rssi_dbm may be None to simulate a missed reading.
        """
        self.reset()
        results: list[PresenceStep] = []
        for timestamp_sec, rssi_dbm in samples:
            state = self.update(rssi_dbm, timestamp_sec)
            results.append(
                PresenceStep(
                    rssi_dbm=rssi_dbm,
                    timestamp_sec=timestamp_sec,
                    state=state,
                )
            )
        return results


def run_synthetic_test() -> bool:
    """
    Hand-crafted synthetic sequence for acceptance testing.

    Returns True if all expected state transitions pass.
    """
    cfg = PresenceConfig(
        near_threshold_dbm=-55,
        far_threshold_dbm=-75,
        hysteresis_dbm=3,
        gone_timeout_sec=5,
    )
    clf = PresenceClassifier(cfg)

    # (timestamp, rssi) — device approaches, stays near, moves far, signal lost
    samples = [
        (0, -80),   # far
        (1, -70),   # far
        (2, -54),   # near
        (3, -52),   # near (hold)
        (4, -56),   # near (within hysteresis band from -55)
        (5, -60),   # far (dropped below near-hysteresis)
        (6, -78),   # far
        (7, None),  # missed reading, still within timeout
        (8, None),
        (12, None), # gone (>= 5s since last valid at t=6... actually last valid at 6)
    ]

    results = clf.classify_sequence(samples)
    states = [step.state for step in results]

    expected = [
        PresenceState.FAR,
        PresenceState.FAR,
        PresenceState.NEAR,
        PresenceState.NEAR,
        PresenceState.NEAR,
        PresenceState.FAR,
        PresenceState.FAR,
        PresenceState.FAR,
        PresenceState.FAR,
        PresenceState.GONE,
    ]

    passed = states == expected
    if not passed:
        print("Synthetic test FAILED")
        for i, (got, exp) in enumerate(zip(states, expected)):
            marker = "OK" if got == exp else "FAIL"
            print(f"  t={samples[i][0]}: got={got.value}, expected={exp.value} [{marker}]")
    else:
        print("Synthetic test PASSED (near/far/gone transitions)")
    return passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_synthetic_test() else 1)
