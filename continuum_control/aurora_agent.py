from __future__ import annotations

import math
import time
from typing import Any

from .config import EMConfig
from .em_core import EMError, EMSample


class AuroraError(RuntimeError):
    """Raised when the Aurora tracker cannot be opened or read."""


_MISSING = object()


class AuroraAgent:
    def __init__(self, config: EMConfig, tracker: Any | None = None, clock=time.monotonic):
        self.config = config
        self.tracker = tracker if tracker is not None else self._make_tracker(config)
        self.clock = clock
        self._started = False

    def _make_tracker(self, config: EMConfig):
        try:
            from sksurgerynditracker.nditracker import NDITracker
        except ModuleNotFoundError as exc:
            raise AuroraError(
                "scikit-surgerynditracker is required for Aurora: "
                "pip install scikit-surgerynditracker"
            ) from exc

        settings: dict[str, object] = {"tracker type": "aurora"}
        if config.serial_port == "auto":
            settings["ports to probe"] = config.ports_to_probe
        else:
            settings["serial port"] = config.serial_port
        return NDITracker(settings)

    def start(self) -> None:
        self.tracker.start_tracking()
        self._started = True

    def close(self) -> None:
        if self._started:
            try:
                self.tracker.stop_tracking()
            finally:
                self.tracker.close()
                self._started = False
            return
        self.tracker.close()

    def read_samples(self) -> dict[str, EMSample]:
        try:
            _, _, _, tracking, quality = self.tracker.get_frame()
        except (TypeError, ValueError) as exc:
            raise AuroraError("NDITracker.get_frame returned an unexpected frame") from exc

        if tracking is None:
            raise AuroraError("NDITracker.get_frame returned no tracking list")

        now_s = float(self.clock())
        samples: dict[str, EMSample] = {}
        for sensor in self.config.sensors:
            raw_transform = _index(tracking, sensor.tool_index, missing=_MISSING)
            if raw_transform is _MISSING:
                raise AuroraError(
                    f"configured sensor {sensor.name} tool_index {sensor.tool_index} "
                    "is missing from tracking list"
                )

            sample_quality = _optional_number(_index(quality, sensor.tool_index, None))
            if raw_transform is None:
                samples[sensor.role] = EMSample(
                    sensor.role,
                    sensor.name,
                    sensor.tool_index,
                    now_s,
                    False,
                    None,
                    sample_quality,
                    "invalid transform",
                )
                continue

            try:
                samples[sensor.role] = EMSample(
                    sensor.role,
                    sensor.name,
                    sensor.tool_index,
                    now_s,
                    True,
                    raw_transform,
                    sample_quality,
                )
            except EMError as exc:
                samples[sensor.role] = EMSample(
                    sensor.role,
                    sensor.name,
                    sensor.tool_index,
                    now_s,
                    False,
                    None,
                    sample_quality,
                    f"invalid transform: {exc}",
                )
        return samples


def _index(values: Any, index: int, missing: Any) -> Any:
    try:
        return values[index]
    except (IndexError, TypeError):
        return missing


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
