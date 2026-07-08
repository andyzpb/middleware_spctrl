from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable, Iterable

from .aurora_agent import AuroraAgent, AuroraError
from .config import ConfigError, EMConfig, load_em_config, load_robot_config
from .core import RobotConfig, SafetyError, bending_deg_to_tick
from .dxl_agent import (
    ADDR_GOAL_POSITION,
    ADDR_HARDWARE_ERROR_STATUS,
    ADDR_OPERATING_MODE,
    ADDR_PRESENT_POSITION,
    ADDR_TORQUE_ENABLE,
    DxlError,
    SdkBackend,
    XL430_W250_MODEL_NUMBER,
)
from .em_core import EMError, EMSample, fresh_samples, relative_transform


ROTATION_LIMIT_TICK = 100
BENDING_LIMIT_DEG = 10.0


class CalibrationError(ValueError):
    """Raised when calibration inputs or samples are invalid."""


@dataclass(frozen=True)
class CalibrationPoint:
    rotation_offset_tick: int
    bending_deg: float


@dataclass(frozen=True)
class CalibrationSampleStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    valid_count: int
    sample_count: int


@dataclass(frozen=True)
class CalibrationSettings:
    rotation_offsets: tuple[int, ...] = (-100, 0, 100)
    bending_degs: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)
    settle_s: float = 1.0
    sample_s: float = 1.0
    min_valid: int = 5


def build_grid(
    rotation_offsets: Iterable[int] = (-100, 0, 100),
    bending_degs: Iterable[float] = (-10.0, -5.0, 0.0, 5.0, 10.0),
) -> list[CalibrationPoint]:
    grid = []
    for rotation_offset in rotation_offsets:
        rotation_offset = int(rotation_offset)
        if abs(rotation_offset) > ROTATION_LIMIT_TICK:
            raise CalibrationError(f"rotation_offset_tick {rotation_offset} outside [-100, 100]")
        for bending_deg in bending_degs:
            bending_deg = float(bending_deg)
            if abs(bending_deg) > BENDING_LIMIT_DEG:
                raise CalibrationError(f"bending_deg {bending_deg} outside [-10.0, 10.0]")
            grid.append(CalibrationPoint(rotation_offset, bending_deg))
    return grid


def summarize_positions(positions: list[tuple[float, float, float]]) -> CalibrationSampleStats:
    if not positions:
        raise CalibrationError("not enough valid EM samples: 0")
    count = len(positions)
    mean = tuple(sum(p[i] for p in positions) / count for i in range(3))
    variance = tuple(sum((p[i] - mean[i]) ** 2 for p in positions) / count for i in range(3))
    std = tuple(v ** 0.5 for v in variance)
    return CalibrationSampleStats(mean=mean, std=std, valid_count=count, sample_count=count)


def csv_fieldnames() -> list[str]:
    return [
        "timestamp",
        "rotation_offset_tick",
        "rotation_goal_tick",
        "rotation_present_tick",
        "bending_deg",
        "bending_goal_tick",
        "bending_present_tick",
        "tip_x_mm_mean",
        "tip_y_mm_mean",
        "tip_z_mm_mean",
        "tip_x_mm_std",
        "tip_y_mm_std",
        "tip_z_mm_std",
        "valid_count",
        "sample_count",
        "abort_reason",
    ]


def run_calibration(
    *,
    robot_config: RobotConfig,
    em_config: EMConfig,
    motor_backend,
    em_agent,
    output_path: str | Path,
    settings: CalibrationSettings = CalibrationSettings(),
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    rotation = robot_config.axes["rotation"]
    bending = robot_config.bending
    grid = build_grid(settings.rotation_offsets, settings.bending_degs)
    sample_count = max(1, int(settings.sample_s * em_config.sample_hz))
    output_path = Path(output_path)
    cleanup_home = False

    motor_backend.open()
    em_agent.start()
    try:
        _validate_motor(motor_backend, rotation.dxl_id, rotation.required_mode)
        _validate_motor(motor_backend, bending.dxl_id, bending.required_mode)
        bending_present = motor_backend.read4(bending.dxl_id, ADDR_PRESENT_POSITION)
        if not bending.min_tick <= bending_present <= bending.max_tick:
            raise CalibrationError(
                f"ID018 present position {bending_present} outside [{bending.min_tick}, {bending.max_tick}]"
            )
        cleanup_home = True

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
            writer.writeheader()
            for point in grid:
                rotation_goal = rotation.home_tick + point.rotation_offset_tick
                bending_goal = bending_deg_to_tick(point.bending_deg, robot_config)
                _write_goal(motor_backend, rotation.dxl_id, rotation_goal)
                _write_goal(motor_backend, bending.dxl_id, bending_goal)
                sleep(settings.settle_s)

                abort_reason = ""
                positions, attempted = _sample_tip_positions(
                    em_agent,
                    sample_count=sample_count,
                    timeout_s=em_config.timeout_s,
                    sample_period_s=1.0 / em_config.sample_hz,
                    sleep=sleep,
                )
                if len(positions) < settings.min_valid:
                    abort_reason = f"not enough valid EM samples: {len(positions)}"
                    stats = CalibrationSampleStats((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), len(positions), attempted)
                else:
                    stats = summarize_positions(positions)
                    stats = CalibrationSampleStats(stats.mean, stats.std, stats.valid_count, attempted)

                writer.writerow(
                    _row(
                        point=point,
                        rotation_goal=rotation_goal,
                        rotation_present=motor_backend.read4(rotation.dxl_id, ADDR_PRESENT_POSITION),
                        bending_goal=bending_goal,
                        bending_present=motor_backend.read4(bending.dxl_id, ADDR_PRESENT_POSITION),
                        stats=stats,
                        abort_reason=abort_reason,
                    )
                )
                if abort_reason:
                    break
    finally:
        try:
            if cleanup_home:
                _write_goal(motor_backend, rotation.dxl_id, rotation.home_tick)
                _write_goal(motor_backend, bending.dxl_id, bending.home_tick)
        finally:
            em_agent.close()
            motor_backend.close()
    return output_path


def _validate_motor(backend, dxl_id: int, required_mode: int) -> None:
    model = backend.ping(dxl_id)
    if model != XL430_W250_MODEL_NUMBER:
        raise CalibrationError(f"ID{dxl_id:03d} model {model} is not XL430-W250")
    mode = backend.read1(dxl_id, ADDR_OPERATING_MODE)
    if mode != required_mode:
        raise CalibrationError(f"ID{dxl_id:03d} mode {mode}; required {required_mode}")
    hardware_error = backend.read1(dxl_id, ADDR_HARDWARE_ERROR_STATUS)
    if hardware_error != 0:
        raise CalibrationError(f"ID{dxl_id:03d} hardware_error={hardware_error}")


def _write_goal(backend, dxl_id: int, tick: int) -> None:
    backend.write1(dxl_id, ADDR_TORQUE_ENABLE, 1)
    backend.write4(dxl_id, ADDR_GOAL_POSITION, int(tick))


def _sample_tip_positions(
    em_agent,
    *,
    sample_count: int,
    timeout_s: float,
    sample_period_s: float,
    sleep: Callable[[float], None],
) -> tuple[list[tuple[float, float, float]], int]:
    positions = []
    for index in range(sample_count):
        samples: dict[str, EMSample] = em_agent.read_samples()
        try:
            pair = fresh_samples(samples, ("tip", "base"), time.monotonic(), timeout_s)
            rel = relative_transform(pair["base"], pair["tip"])
            positions.append((rel[0][3], rel[1][3], rel[2][3]))
        except EMError:
            pass
        if index < sample_count - 1:
            sleep(sample_period_s)
    return positions, sample_count


def _row(
    *,
    point: CalibrationPoint,
    rotation_goal: int,
    rotation_present: int,
    bending_goal: int,
    bending_present: int,
    stats: CalibrationSampleStats,
    abort_reason: str,
) -> dict[str, object]:
    return {
        "timestamp": f"{time.time():.6f}",
        "rotation_offset_tick": point.rotation_offset_tick,
        "rotation_goal_tick": rotation_goal,
        "rotation_present_tick": rotation_present,
        "bending_deg": f"{point.bending_deg:.3f}",
        "bending_goal_tick": bending_goal,
        "bending_present_tick": bending_present,
        "tip_x_mm_mean": f"{stats.mean[0]:.6f}",
        "tip_y_mm_mean": f"{stats.mean[1]:.6f}",
        "tip_z_mm_mean": f"{stats.mean[2]:.6f}",
        "tip_x_mm_std": f"{stats.std[0]:.6f}",
        "tip_y_mm_std": f"{stats.std[1]:.6f}",
        "tip_z_mm_std": f"{stats.std[2]:.6f}",
        "valid_count": stats.valid_count,
        "sample_count": stats.sample_count,
        "abort_reason": abort_reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibrate-em")
    parser.add_argument("--robot-config", default="config/robot.yaml")
    parser.add_argument("--em-config", default="config/em.yaml")
    parser.add_argument("--motor-port", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        robot_config = load_robot_config(args.robot_config)
        em_config = load_em_config(args.em_config)
        motor_backend = SdkBackend(args.motor_port, robot_config.baudrate)
        em_agent = AuroraAgent(em_config)
        output = run_calibration(
            robot_config=robot_config,
            em_config=em_config,
            motor_backend=motor_backend,
            em_agent=em_agent,
            output_path=args.output,
        )
        print(f"calibration written: {output}")
        return 0
    except (AuroraError, CalibrationError, ConfigError, DxlError, SafetyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
