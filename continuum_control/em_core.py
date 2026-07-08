from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


Transform4 = tuple[tuple[float, float, float, float], ...]


class EMError(ValueError):
    """Raised when EM samples are missing, stale, or invalid."""


def coerce_transform4(value: object, context: str = "transform") -> Transform4:
    try:
        rows = tuple(tuple(float(cell) for cell in row) for row in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise EMError(f"{context} must be a 4x4 transform") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise EMError(f"{context} must be a 4x4 transform")
    if any(not math.isfinite(cell) for row in rows for cell in row):
        raise EMError(f"{context} must contain only finite values")
    return rows


@dataclass(frozen=True)
class EMSample:
    role: str
    name: str
    tool_index: int
    timestamp_s: float
    valid: bool
    transform: Transform4 | None
    quality: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.role:
            raise EMError("sample role must be non-empty")
        if not self.name:
            raise EMError("sample name must be non-empty")
        if self.tool_index < 0:
            raise EMError("sample tool_index must be >= 0")
        if not math.isfinite(float(self.timestamp_s)):
            raise EMError("sample timestamp_s must be finite")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        if self.valid:
            if self.transform is None:
                raise EMError(f"{self.role} valid sample missing transform")
            object.__setattr__(self, "transform", coerce_transform4(self.transform, f"{self.role} transform"))

    @property
    def position_mm(self) -> tuple[float, float, float]:
        if not self.valid or self.transform is None:
            raise EMError(f"{self.role} invalid: {self.error or 'missing transform'}")
        return (self.transform[0][3], self.transform[1][3], self.transform[2][3])


def fresh_samples(
    samples_by_role: dict[str, EMSample],
    required_roles: Iterable[str],
    now_s: float,
    timeout_s: float,
) -> dict[str, EMSample]:
    fresh: dict[str, EMSample] = {}
    for role in required_roles:
        if role not in samples_by_role:
            raise EMError(f"missing {role}")
        sample = samples_by_role[role]
        if not sample.valid:
            raise EMError(f"{role} invalid: {sample.error or 'missing transform'}")
        age_s = float(now_s) - sample.timestamp_s
        if age_s > timeout_s:
            raise EMError(f"{role} stale: age_s={age_s:.3f} timeout_s={timeout_s:.3f}")
        fresh[role] = sample
    return fresh


def relative_transform(base: EMSample, moving: EMSample) -> Transform4:
    if not base.valid or base.transform is None:
        raise EMError(f"{base.role} invalid: {base.error or 'missing transform'}")
    if not moving.valid or moving.transform is None:
        raise EMError(f"{moving.role} invalid: {moving.error or 'missing transform'}")

    rb = _rotation(base.transform)
    rt = _transpose3(rb)
    rm = _rotation(moving.transform)
    pb = base.position_mm
    pm = moving.position_mm
    rel_r = _matmul3(rt, rm)
    rel_p = _matvec3(rt, (pm[0] - pb[0], pm[1] - pb[1], pm[2] - pb[2]))
    return (
        (rel_r[0][0], rel_r[0][1], rel_r[0][2], rel_p[0]),
        (rel_r[1][0], rel_r[1][1], rel_r[1][2], rel_p[1]),
        (rel_r[2][0], rel_r[2][1], rel_r[2][2], rel_p[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation(t: Transform4) -> tuple[tuple[float, float, float], ...]:
    return (
        (t[0][0], t[0][1], t[0][2]),
        (t[1][0], t[1][1], t[1][2]),
        (t[2][0], t[2][1], t[2][2]),
    )


def _transpose3(m: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def _matmul3(
    a: tuple[tuple[float, float, float], ...],
    b: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _matvec3(
    m: tuple[tuple[float, float, float], ...],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))
