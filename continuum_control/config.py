from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import AxisConfig, RobotConfig


class ConfigError(ValueError):
    """Raised when robot configuration is missing or invalid."""


REQUIRED_ROLES = ("translation", "rotation", "bending")


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must be a mapping")
    return data


def _require_int(data: dict[str, Any], key: str, context: str) -> int:
    if key not in data:
        raise ConfigError(f"{context} missing {key}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context}.{key} must be an integer")
    return value


def _optional_float(data: dict[str, Any], key: str, context: str) -> float | None:
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context}.{key} must be a number")
    return float(value)


def _axis_from_mapping(role: str, data: Any) -> AxisConfig:
    values = _require_mapping(data, f"motors.{role}")
    return AxisConfig(
        role=role,
        dxl_id=_require_int(values, "id", f"motors.{role}"),
        required_mode=_require_int(values, "required_mode", f"motors.{role}"),
        home_tick=_require_int(values, "home_tick", f"motors.{role}"),
        min_tick=_require_int(values, "min_tick", f"motors.{role}"),
        max_tick=_require_int(values, "max_tick", f"motors.{role}"),
        profile_velocity=_require_int(values, "profile_velocity", f"motors.{role}"),
        profile_acceleration=_require_int(values, "profile_acceleration", f"motors.{role}"),
        logical_min_deg=_optional_float(values, "logical_min_deg", f"motors.{role}"),
        logical_max_deg=_optional_float(values, "logical_max_deg", f"motors.{role}"),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigError("PyYAML is required: pip install PyYAML") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _require_mapping(data, str(path))


def load_robot_config(path: str | Path) -> RobotConfig:
    data = _load_yaml(Path(path))
    serial_port = data.get("serial_port")
    if not isinstance(serial_port, str) or not serial_port:
        raise ConfigError("serial_port must be a non-empty string")

    baudrate = data.get("baudrate")
    if isinstance(baudrate, bool) or not isinstance(baudrate, int):
        raise ConfigError("baudrate must be an integer")

    bus_watchdog = data.get("bus_watchdog", 25)
    if isinstance(bus_watchdog, bool) or not isinstance(bus_watchdog, int):
        raise ConfigError("bus_watchdog must be an integer")
    if bus_watchdog < 0 or bus_watchdog > 127:
        raise ConfigError("bus_watchdog must be in 0..127")

    motors = _require_mapping(data.get("motors"), "motors")
    axes = {}
    for role in REQUIRED_ROLES:
        if role not in motors:
            raise ConfigError(f"motors missing {role}")
        axes[role] = _axis_from_mapping(role, motors[role])

    ids = [axis.dxl_id for axis in axes.values()]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"motor IDs must be unique: {ids}")

    bending = axes["bending"]
    if bending.required_mode != 3:
        raise ConfigError("bending.required_mode must be 3")
    if (bending.min_tick, bending.home_tick, bending.max_tick) != (1195, 1536, 1877):
        raise ConfigError("bending ticks must be min=1195, home=1536, max=1877")
    if (bending.logical_min_deg, bending.logical_max_deg) != (-20.0, 20.0):
        raise ConfigError("bending logical range must be -20.0..20.0")

    return RobotConfig(
        serial_port=serial_port,
        baudrate=baudrate,
        axes=axes,
        bus_watchdog=bus_watchdog,
    )
