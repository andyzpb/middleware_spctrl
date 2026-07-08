from __future__ import annotations

import glob
import re
import sys


class SerialPortError(RuntimeError):
    """Raised when U2D2 serial-port selection is ambiguous or missing."""


def _list_system_ports(platform: str) -> list[str]:
    try:
        from serial.tools import list_ports
    except ModuleNotFoundError:
        list_ports = None

    if list_ports is not None:
        return sorted(port.device for port in list_ports.comports())

    if platform == "darwin":
        return sorted(glob.glob("/dev/tty.usbserial-*") + glob.glob("/dev/cu.usbserial-*"))
    if platform.startswith("linux"):
        return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return []


def _matches_platform_port(port: str, platform: str) -> bool:
    if platform == "darwin":
        return port.startswith("/dev/tty.usbserial-") or port.startswith("/dev/cu.usbserial-")
    if platform.startswith("linux"):
        return port.startswith("/dev/ttyUSB") or port.startswith("/dev/ttyACM")
    if platform.startswith("win"):
        return re.fullmatch(r"COM[1-9][0-9]*", port, re.IGNORECASE) is not None
    return False


def resolve_serial_port(
    config_port: str,
    platform: str | None = None,
    candidates: list[str] | None = None,
) -> str:
    if config_port != "auto":
        return config_port

    platform = platform or sys.platform
    ports = list(candidates) if candidates is not None else _list_system_ports(platform)
    matches = [port for port in ports if _matches_platform_port(port, platform)]

    if not matches:
        raise SerialPortError(f"no U2D2-like serial port found for platform {platform}")
    if len(matches) > 1:
        raise SerialPortError(f"multiple U2D2-like serial ports found: {matches}")
    return matches[0]
