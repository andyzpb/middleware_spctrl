from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import (
    MotorGoal,
    POSITION_CONTROL_MODE,
    RobotConfig,
    SafetyError,
    SafetyGate,
    Command,
)


XL430_W250_MODEL_NUMBER = 1060

ADDR_OPERATING_MODE = 11
ADDR_MIN_POSITION_LIMIT = 52
ADDR_MAX_POSITION_LIMIT = 48
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_BUS_WATCHDOG = 98
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_MOVING = 122
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

TORQUE_OFF = 0
TORQUE_ON = 1


class DxlError(RuntimeError):
    """Raised when the DYNAMIXEL bus or a motor violates the hardware contract."""


class DxlBackend(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def ping(self, dxl_id: int) -> int | None: ...
    def read1(self, dxl_id: int, address: int) -> int: ...
    def read2(self, dxl_id: int, address: int) -> int: ...
    def read4(self, dxl_id: int, address: int) -> int: ...
    def write1(self, dxl_id: int, address: int, value: int) -> None: ...
    def write2(self, dxl_id: int, address: int, value: int) -> None: ...
    def write4(self, dxl_id: int, address: int, value: int) -> None: ...


@dataclass(frozen=True)
class MotorStatus:
    dxl_id: int
    operating_mode: int
    torque_enabled: bool
    present_position: int
    hardware_error_status: int
    input_voltage_raw: int
    temperature_c: int


class SdkBackend:
    def __init__(self, port: str, baudrate: int, protocol_version: float = 2.0):
        try:
            from dynamixel_sdk import PacketHandler, PortHandler
        except ModuleNotFoundError as exc:
            raise DxlError("dynamixel_sdk is required: pip install dynamixel-sdk") from exc

        self.port_name = port
        self.baudrate = baudrate
        self.port = PortHandler(port)
        self.packet = PacketHandler(protocol_version)

    def open(self) -> None:
        if not self.port.openPort():
            raise DxlError(f"failed to open DYNAMIXEL port {self.port_name}")
        if not self.port.setBaudRate(self.baudrate):
            raise DxlError(f"failed to set baudrate {self.baudrate} on {self.port_name}")

    def close(self) -> None:
        self.port.closePort()

    def _check(self, result: int, error: int, dxl_id: int, op: str, address: int | None = None) -> None:
        if result != 0:
            raise DxlError(f"[ID:{dxl_id}] {op} failed: {self.packet.getTxRxResult(result)}")
        if error != 0:
            where = "" if address is None else f" addr={address}"
            raise DxlError(f"[ID:{dxl_id}] {op}{where} packet error: {self.packet.getRxPacketError(error)}")

    def ping(self, dxl_id: int) -> int | None:
        model, result, error = self.packet.ping(self.port, dxl_id)
        if result != 0:
            return None
        if error != 0:
            raise DxlError(f"[ID:{dxl_id}] ping packet error: {self.packet.getRxPacketError(error)}")
        return int(model)

    def read1(self, dxl_id: int, address: int) -> int:
        value, result, error = self.packet.read1ByteTxRx(self.port, dxl_id, address)
        self._check(result, error, dxl_id, "read1", address)
        return int(value)

    def read2(self, dxl_id: int, address: int) -> int:
        value, result, error = self.packet.read2ByteTxRx(self.port, dxl_id, address)
        self._check(result, error, dxl_id, "read2", address)
        return int(value)

    def read4(self, dxl_id: int, address: int) -> int:
        value, result, error = self.packet.read4ByteTxRx(self.port, dxl_id, address)
        self._check(result, error, dxl_id, "read4", address)
        return int(value)

    def write1(self, dxl_id: int, address: int, value: int) -> None:
        result, error = self.packet.write1ByteTxRx(self.port, dxl_id, address, int(value))
        self._check(result, error, dxl_id, "write1", address)

    def write2(self, dxl_id: int, address: int, value: int) -> None:
        result, error = self.packet.write2ByteTxRx(self.port, dxl_id, address, int(value))
        self._check(result, error, dxl_id, "write2", address)

    def write4(self, dxl_id: int, address: int, value: int) -> None:
        result, error = self.packet.write4ByteTxRx(self.port, dxl_id, address, int(value))
        self._check(result, error, dxl_id, "write4", address)


class DxlAgent:
    def __init__(self, config: RobotConfig, backend: DxlBackend | None = None):
        self.config = config
        self.gate = SafetyGate(config)
        self.backend = backend or SdkBackend(config.serial_port, config.baudrate)
        self.armed = False

    def open(self) -> None:
        self.backend.open()

    def close(self) -> None:
        self.backend.close()

    def status(self) -> list[MotorStatus]:
        rows = []
        for axis in self.config.axes.values():
            dxl_id = axis.dxl_id
            rows.append(
                MotorStatus(
                    dxl_id=dxl_id,
                    operating_mode=self.backend.read1(dxl_id, ADDR_OPERATING_MODE),
                    torque_enabled=bool(self.backend.read1(dxl_id, ADDR_TORQUE_ENABLE)),
                    present_position=self.backend.read4(dxl_id, ADDR_PRESENT_POSITION),
                    hardware_error_status=self.backend.read1(dxl_id, ADDR_HARDWARE_ERROR_STATUS),
                    input_voltage_raw=self.backend.read2(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE),
                    temperature_c=self.backend.read1(dxl_id, ADDR_PRESENT_TEMPERATURE),
                )
            )
        return rows

    def arm(self) -> None:
        self.backend.open()
        self._verify_ids_and_models()
        for axis in self.config.axes.values():
            dxl_id = axis.dxl_id
            self.backend.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
            if self.backend.read1(dxl_id, ADDR_OPERATING_MODE) != axis.required_mode:
                self.backend.write1(dxl_id, ADDR_OPERATING_MODE, axis.required_mode)
            if self.backend.read1(dxl_id, ADDR_OPERATING_MODE) != axis.required_mode:
                raise DxlError(f"ID {dxl_id} operating mode did not change to {axis.required_mode}")
            if axis.required_mode == POSITION_CONTROL_MODE:
                self.backend.write4(dxl_id, ADDR_MIN_POSITION_LIMIT, axis.min_tick)
                self.backend.write4(dxl_id, ADDR_MAX_POSITION_LIMIT, axis.max_tick)
                if self.backend.read4(dxl_id, ADDR_MIN_POSITION_LIMIT) != axis.min_tick:
                    raise DxlError(f"ID {dxl_id} min position limit did not change to {axis.min_tick}")
                if self.backend.read4(dxl_id, ADDR_MAX_POSITION_LIMIT) != axis.max_tick:
                    raise DxlError(f"ID {dxl_id} max position limit did not change to {axis.max_tick}")
            self.backend.write4(dxl_id, ADDR_PROFILE_VELOCITY, axis.profile_velocity)
            self.backend.write4(dxl_id, ADDR_PROFILE_ACCELERATION, axis.profile_acceleration)
            self.backend.write1(dxl_id, ADDR_BUS_WATCHDOG, self.config.bus_watchdog)
            present = self.backend.read4(dxl_id, ADDR_PRESENT_POSITION)
            if axis.role == "bending" and not axis.min_tick <= present <= axis.max_tick:
                raise SafetyError(
                    f"ID018 present position {present} outside [{axis.min_tick}, {axis.max_tick}]"
                )
        for axis in self.config.axes.values():
            self.backend.write1(axis.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        self.armed = True

    def disarm(self) -> None:
        for axis in self.config.axes.values():
            self.backend.write1(axis.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        self.armed = False

    def home(self) -> MotorGoal:
        self._require_armed()
        goal = self.gate.home_goal()
        self.write_goal(goal)
        return goal

    def jog_bending(self, deg: float) -> MotorGoal:
        self._require_armed()
        goal = self.gate.command_to_goals(Command(bending_deg=deg, source="operator"))
        self.write_goal(goal)
        return goal

    def write_goal(self, goal: MotorGoal) -> None:
        self._require_armed()
        for dxl_id, tick in goal.ticks.items():
            axis = self._axis_for_id(dxl_id)
            if not axis.min_tick <= int(tick) <= axis.max_tick:
                raise SafetyError(f"ID{dxl_id:03d} goal {tick} outside [{axis.min_tick}, {axis.max_tick}]")
            if axis.role == "bending" and self.backend.read1(dxl_id, ADDR_OPERATING_MODE) != POSITION_CONTROL_MODE:
                raise SafetyError("ID018 must be in Position Control Mode before publishing goals")
            self.backend.write4(dxl_id, ADDR_GOAL_POSITION, int(tick))

    def _verify_ids_and_models(self) -> None:
        for axis in self.config.axes.values():
            model = self.backend.ping(axis.dxl_id)
            if model is None:
                raise DxlError(f"missing DYNAMIXEL ID {axis.dxl_id}")
            if model != XL430_W250_MODEL_NUMBER:
                raise DxlError(f"ID {axis.dxl_id} model {model} is not XL430-W250 ({XL430_W250_MODEL_NUMBER})")

    def _axis_for_id(self, dxl_id: int):
        for axis in self.config.axes.values():
            if axis.dxl_id == dxl_id:
                return axis
        raise SafetyError(f"unknown DYNAMIXEL ID {dxl_id}")

    def _require_armed(self) -> None:
        if not self.armed:
            raise DxlError("agent is not armed")
