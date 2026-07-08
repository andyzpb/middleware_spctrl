from __future__ import annotations

from dataclasses import dataclass
import math


POSITION_CONTROL_MODE = 3


class SafetyError(ValueError):
    """Raised when a command or hardware state violates the robot contract."""


@dataclass(frozen=True)
class AxisConfig:
    role: str
    dxl_id: int
    required_mode: int
    home_tick: int
    min_tick: int
    max_tick: int
    profile_velocity: int
    profile_acceleration: int
    logical_min_deg: float | None = None
    logical_max_deg: float | None = None


@dataclass(frozen=True)
class RobotConfig:
    serial_port: str
    baudrate: int
    axes: dict[str, AxisConfig]
    bus_watchdog: int = 25

    @property
    def bending(self) -> AxisConfig:
        return self.axes["bending"]

    @property
    def required_ids(self) -> set[int]:
        return {axis.dxl_id for axis in self.axes.values()}


@dataclass(frozen=True)
class Command:
    translation_tick: int | None = None
    rotation_tick: int | None = None
    bending_deg: float | None = None
    source: str = "operator"


@dataclass(frozen=True)
class MotorGoal:
    ticks: dict[int, int]
    saturated: bool = False


def default_robot_config() -> RobotConfig:
    return RobotConfig(
        serial_port="auto",
        baudrate=1_000_000,
        bus_watchdog=25,
        axes={
            "translation": AxisConfig(
                role="translation",
                dxl_id=8,
                required_mode=POSITION_CONTROL_MODE,
                home_tick=2048,
                min_tick=0,
                max_tick=4095,
                profile_velocity=30,
                profile_acceleration=10,
            ),
            "rotation": AxisConfig(
                role="rotation",
                dxl_id=12,
                required_mode=POSITION_CONTROL_MODE,
                home_tick=2048,
                min_tick=0,
                max_tick=4095,
                profile_velocity=22,
                profile_acceleration=10,
            ),
            "bending": AxisConfig(
                role="bending",
                dxl_id=18,
                required_mode=POSITION_CONTROL_MODE,
                home_tick=1652,
                min_tick=1368,
                max_tick=1936,
                profile_velocity=13,
                profile_acceleration=6,
                logical_min_deg=-25.0,
                logical_max_deg=25.0,
            ),
        },
    )


def _require_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SafetyError(f"{name} must be finite")
    return value


def _bending_limits(config: RobotConfig) -> tuple[AxisConfig, float, float]:
    axis = config.bending
    if axis.logical_min_deg is None or axis.logical_max_deg is None:
        raise SafetyError("bending axis is missing logical degree limits")
    return axis, float(axis.logical_min_deg), float(axis.logical_max_deg)


def bending_deg_to_tick(deg: float, config: RobotConfig | None = None) -> int:
    config = config or default_robot_config()
    axis, lo_deg, hi_deg = _bending_limits(config)
    deg = _require_finite(deg, "bending_deg")
    if deg < lo_deg or deg > hi_deg:
        raise SafetyError(f"bending_deg {deg} outside [{lo_deg}, {hi_deg}]")

    if deg <= 0.0:
        tick = round(axis.home_tick + deg * (axis.home_tick - axis.min_tick) / abs(lo_deg))
    else:
        tick = round(axis.home_tick + deg * (axis.max_tick - axis.home_tick) / hi_deg)
    return int(tick)


def bending_tick_to_deg(tick: int, config: RobotConfig | None = None) -> float:
    config = config or default_robot_config()
    axis, lo_deg, hi_deg = _bending_limits(config)
    tick = int(tick)
    if tick < axis.min_tick or tick > axis.max_tick:
        raise SafetyError(f"bending tick {tick} outside [{axis.min_tick}, {axis.max_tick}]")

    if tick <= axis.home_tick:
        return (tick - axis.home_tick) * abs(lo_deg) / (axis.home_tick - axis.min_tick)
    return (tick - axis.home_tick) * hi_deg / (axis.max_tick - axis.home_tick)


def clamp_tick(tick: int, axis: AxisConfig) -> tuple[int, bool]:
    tick = int(tick)
    clamped = max(axis.min_tick, min(axis.max_tick, tick))
    return clamped, clamped != tick


class SafetyGate:
    def __init__(self, config: RobotConfig):
        self.config = config

    def home_goal(self) -> MotorGoal:
        return MotorGoal({axis.dxl_id: axis.home_tick for axis in self.config.axes.values()})

    def command_to_goals(self, command: Command) -> MotorGoal:
        ticks: dict[int, int] = {}
        saturated = False

        if command.translation_tick is not None:
            axis = self.config.axes["translation"]
            tick, was_saturated = clamp_tick(command.translation_tick, axis)
            if was_saturated and command.source != "controller":
                raise SafetyError("translation_tick outside configured limits")
            ticks[axis.dxl_id] = tick
            saturated = saturated or was_saturated

        if command.rotation_tick is not None:
            axis = self.config.axes["rotation"]
            tick, was_saturated = clamp_tick(command.rotation_tick, axis)
            if was_saturated and command.source != "controller":
                raise SafetyError("rotation_tick outside configured limits")
            ticks[axis.dxl_id] = tick
            saturated = saturated or was_saturated

        if command.bending_deg is not None:
            axis, lo_deg, hi_deg = _bending_limits(self.config)
            requested = _require_finite(command.bending_deg, "bending_deg")
            clamped_deg = max(lo_deg, min(hi_deg, requested))
            was_saturated = clamped_deg != requested
            if was_saturated and command.source != "controller":
                raise SafetyError(f"bending_deg {requested} outside [{lo_deg}, {hi_deg}]")
            ticks[axis.dxl_id] = bending_deg_to_tick(clamped_deg, self.config)
            saturated = saturated or was_saturated

        if not ticks:
            raise SafetyError("command contains no axis goal")

        return MotorGoal(ticks=ticks, saturated=saturated)
