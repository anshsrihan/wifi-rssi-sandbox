"""
Read-only Tier 1 RSSI sources for macOS, Linux, and Windows.

All operations are passive / read-only — no packet injection, no deauth,
no router configuration changes.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class RssiReading:
    """Single RSSI sample from a Tier 1 source."""

    rssi_dbm: float
    label: str
    source_tier: int = 1
    source_method: str = "unknown"
    noise_dbm: Optional[float] = None


class RssiReadError(Exception):
    """Raised when RSSI cannot be read from any available Tier 1 source."""


def detect_platform() -> str:
    """Return normalized platform id: darwin, linux, or windows."""
    system = sys.platform
    if system == "darwin":
        return "darwin"
    if system.startswith("linux"):
        return "linux"
    if system == "windows":
        return "windows"
    return system


def _run_command(args: list[str], timeout: float = 10.0) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RssiReadError(f"Command not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RssiReadError(f"Command timed out: {' '.join(args)}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RssiReadError(f"{' '.join(args)} failed: {detail}")

    return result.stdout


def read_rssi_macos(label: str) -> RssiReading:
    """
    Read laptop RSSI on macOS.

    Primary: CoreWLAN via a short Swift subprocess (~300 ms, no extra pip deps).
    Fallback: system_profiler SPAirPortDataType (slower, ~2–6 s).
    Legacy: airport -I if the binary still exists on older macOS releases.
    """
    swift_code = """
import CoreWLAN
let client = CWWiFiClient.shared()
guard let iface = client.interface() else {
    fputs("ERROR: No WiFi interface\\n", stderr)
    exit(1)
}
print("RSSI:\\(iface.rssiValue())")
if let noise = iface.noiseMeasurement() as NSNumber? {
    print("NOISE:\\(noise.intValue)")
}
"""
    try:
        output = _run_command(["swift", "-e", swift_code], timeout=15.0)
        rssi_match = re.search(r"RSSI:(-?\d+)", output)
        if rssi_match:
            noise = None
            noise_match = re.search(r"NOISE:(-?\d+)", output)
            if noise_match:
                noise = float(noise_match.group(1))
            return RssiReading(
                rssi_dbm=float(rssi_match.group(1)),
                label=label,
                source_method="corewlan_swift",
                noise_dbm=noise,
            )
    except RssiReadError:
        pass

    airport_paths = [
        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
        "/usr/local/bin/airport",
    ]
    for airport_path in airport_paths:
        if not shutil.which(airport_path) and not __import__("os").path.isfile(airport_path):
            continue
        try:
            output = _run_command([airport_path, "-I"], timeout=5.0)
            rssi_match = re.search(r"agrCtlRSSI\s*:\s*(-?\d+)", output)
            if rssi_match:
                noise_match = re.search(r"agrCtlNoise\s*:\s*(-?\d+)", output)
                noise = float(noise_match.group(1)) if noise_match else None
                return RssiReading(
                    rssi_dbm=float(rssi_match.group(1)),
                    label=label,
                    source_method="airport_cli",
                    noise_dbm=noise,
                )
        except RssiReadError:
            continue

    try:
        output = _run_command(["system_profiler", "SPAirPortDataType"], timeout=30.0)
        connected_block = _extract_connected_network_block(output)
        signal_match = re.search(
            r"Signal / Noise:\s*(-?\d+)\s*dBm\s*/\s*(-?\d+)\s*dBm",
            connected_block,
        )
        if signal_match:
            return RssiReading(
                rssi_dbm=float(signal_match.group(1)),
                label=label,
                source_method="system_profiler",
                noise_dbm=float(signal_match.group(2)),
            )
    except RssiReadError as exc:
        raise RssiReadError(
            "Could not read WiFi RSSI on macOS. Ensure Wi-Fi is connected. "
            "Tried CoreWLAN (swift), legacy airport, and system_profiler. "
            f"Last error: {exc}"
        ) from exc

    raise RssiReadError(
        "Could not parse RSSI from macOS Wi-Fi interfaces. "
        "Is your laptop connected to a Wi-Fi network?"
    )


def _extract_connected_network_block(profiler_output: str) -> str:
    """Return the 'Current Network Information' section for a connected interface."""
    lines = profiler_output.splitlines()
    capture = False
    captured: list[str] = []
    indent_level: Optional[int] = None

    for line in lines:
        if "Current Network Information:" in line:
            capture = True
            captured = [line]
            indent_level = len(line) - len(line.lstrip(" "))
            continue

        if capture:
            if line.strip() == "":
                captured.append(line)
                continue
            current_indent = len(line) - len(line.lstrip(" "))
            if indent_level is not None and current_indent <= indent_level and line.strip():
                break
            captured.append(line)

    return "\n".join(captured)


def read_rssi_linux(label: str, interface: Optional[str] = None) -> RssiReading:
    """
    Read RSSI on Linux.

    Tries `iw dev <iface> station dump` when an AP/station view is available,
    otherwise falls back to `nmcli device wifi list` for the connected network.
    """
    if interface is None:
        interface = _detect_linux_wifi_interface()

    if shutil.which("iw"):
        try:
            output = _run_command(["iw", "dev", interface, "link"], timeout=5.0)
            connected_match = re.search(r"Connected to\s+([0-9a-f:]{17})", output, re.I)
            if connected_match:
                station_output = _run_command(
                    ["iw", "dev", interface, "station", "dump"], timeout=5.0
                )
                signal_match = re.search(r"signal:\s*(-?\d+\.?\d*)\s*dBm", station_output)
                if signal_match:
                    return RssiReading(
                        rssi_dbm=float(signal_match.group(1)),
                        label=connected_match.group(1).lower(),
                        source_method="iw_station_dump",
                    )

            signal_match = re.search(r"signal:\s*(-?\d+\.?\d*)\s*dBm", output)
            if signal_match:
                return RssiReading(
                    rssi_dbm=float(signal_match.group(1)),
                    label=label,
                    source_method="iw_link",
                )
        except RssiReadError:
            pass

    if shutil.which("nmcli"):
        try:
            output = _run_command(["nmcli", "-f", "ACTIVE,SSID,SIGNAL", "device", "wifi"], timeout=5.0)
            for line in output.splitlines():
                if not line.strip().startswith("*"):
                    continue
                parts = [part.strip() for part in re.split(r"\s{2,}", line.strip()) if part.strip()]
                if len(parts) >= 3 and parts[-1].isdigit():
                    # nmcli reports 0-100, approximate dBm conversion used by NetworkManager
                    signal_pct = int(parts[-1])
                    approx_dbm = (signal_pct / 2.0) - 100.0
                    ssid = parts[1] if len(parts) > 2 else label
                    return RssiReading(
                        rssi_dbm=approx_dbm,
                        label=ssid,
                        source_method="nmcli_wifi",
                    )
        except RssiReadError:
            pass

    raise RssiReadError(
        "Could not read WiFi RSSI on Linux. Connect to Wi-Fi and ensure "
        "`iw` or `nmcli` is installed. Reading station dump may require "
        "running from an AP (e.g. OpenWrt) or your laptop as the AP."
    )


def _detect_linux_wifi_interface() -> str:
    if shutil.which("iw"):
        try:
            output = _run_command(["iw", "dev"], timeout=5.0)
            match = re.search(r"Interface\s+(\S+)", output)
            if match:
                return match.group(1)
        except RssiReadError:
            pass
    return "wlan0"


def read_rssi_windows(label: str) -> RssiReading:
    """Read laptop RSSI via `netsh wlan show interfaces`."""
    output = _run_command(["netsh", "wlan", "show", "interfaces"], timeout=10.0)
    signal_match = re.search(r"Signal\s*:\s*(\d+)%", output)
    if not signal_match:
        raise RssiReadError(
            "Could not parse RSSI from `netsh wlan show interfaces`. "
            "Is Wi-Fi connected?"
        )

    signal_pct = int(signal_match.group(1))
    approx_dbm = (signal_pct / 2.0) - 100.0
    ssid_match = re.search(r"SSID\s*:\s*(.+)", output)
    ssid = ssid_match.group(1).strip() if ssid_match else label

    return RssiReading(
        rssi_dbm=approx_dbm,
        label=ssid,
        source_method="netsh_wlan",
    )


def read_rssi_tier1(label: str = "laptop", interface: Optional[str] = None) -> RssiReading:
    """
    Read RSSI using the best available Tier 1 source for this platform.

    Raises RssiReadError with a human-readable message on failure.
    """
    plat = detect_platform()
    if plat == "darwin":
        return read_rssi_macos(label)
    if plat == "linux":
        return read_rssi_linux(label, interface=interface)
    if plat == "windows":
        return read_rssi_windows(label)

    raise RssiReadError(f"Unsupported platform for Tier 1 RSSI reading: {platform.system()}")


def describe_tier1_source() -> str:
    """Human-readable description of the Tier 1 source for this machine."""
    plat = detect_platform()
    if plat == "darwin":
        return (
            "Tier 1 / macOS: CoreWLAN via swift (fallback: system_profiler). "
            "Reads RSSI of the Wi-Fi network this laptop is connected to. "
            "No sudo required."
        )
    if plat == "linux":
        return (
            "Tier 1 / Linux: iw station dump or nmcli. "
            "Reads connected-client or local-station RSSI. No packet injection."
        )
    if plat == "windows":
        return (
            "Tier 1 / Windows: netsh wlan show interfaces. "
            "Reads local adapter signal strength."
        )
    return f"Tier 1 / unknown platform: {platform.system()}"
