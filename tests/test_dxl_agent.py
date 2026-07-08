import unittest

from continuum_control.core import SafetyError, default_robot_config
from continuum_control.dxl_agent import (
    ADDR_BUS_WATCHDOG,
    ADDR_GOAL_POSITION,
    ADDR_MAX_POSITION_LIMIT,
    ADDR_MIN_POSITION_LIMIT,
    ADDR_OPERATING_MODE,
    ADDR_PROFILE_ACCELERATION,
    ADDR_PROFILE_VELOCITY,
    ADDR_TORQUE_ENABLE,
    DxlAgent,
    DxlError,
    XL430_W250_MODEL_NUMBER,
)


class FakeBackend:
    def __init__(self):
        self.models = {8: XL430_W250_MODEL_NUMBER, 12: XL430_W250_MODEL_NUMBER, 18: XL430_W250_MODEL_NUMBER}
        self.registers = {}
        self.writes = []
        for dxl_id in self.models:
            self.registers[(dxl_id, ADDR_OPERATING_MODE)] = 0
            self.registers[(dxl_id, ADDR_TORQUE_ENABLE)] = 0
            self.registers[(dxl_id, 70)] = 0
            self.registers[(dxl_id, 132)] = 2048
            self.registers[(dxl_id, 144)] = 111
            self.registers[(dxl_id, 146)] = 25
        self.registers[(18, 132)] = 1652

    def open(self):
        self.writes.append(("open",))

    def close(self):
        self.writes.append(("close",))

    def ping(self, dxl_id):
        return self.models.get(dxl_id)

    def read1(self, dxl_id, address):
        return self.registers[(dxl_id, address)]

    def read2(self, dxl_id, address):
        return self.registers[(dxl_id, address)]

    def read4(self, dxl_id, address):
        return self.registers[(dxl_id, address)]

    def write1(self, dxl_id, address, value):
        self.registers[(dxl_id, address)] = int(value)
        self.writes.append((dxl_id, address, int(value)))

    def write2(self, dxl_id, address, value):
        self.registers[(dxl_id, address)] = int(value)
        self.writes.append((dxl_id, address, int(value)))

    def write4(self, dxl_id, address, value):
        self.registers[(dxl_id, address)] = int(value)
        self.writes.append((dxl_id, address, int(value)))


class DxlAgentTests(unittest.TestCase):
    def test_arm_configures_bending_position_mode_and_limits(self):
        backend = FakeBackend()
        agent = DxlAgent(default_robot_config(), backend)

        agent.arm()

        self.assertIn((18, ADDR_TORQUE_ENABLE, 0), backend.writes)
        self.assertIn((18, ADDR_OPERATING_MODE, 3), backend.writes)
        self.assertIn((18, ADDR_MIN_POSITION_LIMIT, 1368), backend.writes)
        self.assertIn((18, ADDR_MAX_POSITION_LIMIT, 1936), backend.writes)
        self.assertIn((18, ADDR_PROFILE_VELOCITY, 13), backend.writes)
        self.assertIn((18, ADDR_PROFILE_ACCELERATION, 6), backend.writes)
        self.assertIn((18, ADDR_BUS_WATCHDOG, 25), backend.writes)
        self.assertEqual(backend.writes[-1], (18, ADDR_TORQUE_ENABLE, 1))

    def test_arm_rejects_missing_motor_id(self):
        backend = FakeBackend()
        del backend.models[12]
        agent = DxlAgent(default_robot_config(), backend)

        with self.assertRaises(DxlError):
            agent.arm()

    def test_arm_rejects_unsafe_bending_present_position_before_torque_on(self):
        backend = FakeBackend()
        backend.registers[(18, 132)] = 2000
        agent = DxlAgent(default_robot_config(), backend)

        with self.assertRaises(SafetyError):
            agent.arm()

        self.assertNotIn((8, ADDR_TORQUE_ENABLE, 1), backend.writes)
        self.assertNotIn((12, ADDR_TORQUE_ENABLE, 1), backend.writes)
        self.assertNotIn((18, ADDR_TORQUE_ENABLE, 1), backend.writes)

    def test_arm_rejects_mode_that_does_not_stick(self):
        class StickyBadModeBackend(FakeBackend):
            def write1(self, dxl_id, address, value):
                self.writes.append((dxl_id, address, int(value)))
                if dxl_id == 18 and address == ADDR_OPERATING_MODE:
                    return
                self.registers[(dxl_id, address)] = int(value)

        backend = StickyBadModeBackend()
        agent = DxlAgent(default_robot_config(), backend)

        with self.assertRaises(DxlError):
            agent.arm()

        self.assertNotIn((18, ADDR_TORQUE_ENABLE, 1), backend.writes)

    def test_home_requires_arm_then_writes_all_home_ticks(self):
        backend = FakeBackend()
        agent = DxlAgent(default_robot_config(), backend)

        with self.assertRaises(DxlError):
            agent.home()

        agent.arm()
        agent.home()

        self.assertIn((8, ADDR_GOAL_POSITION, 2048), backend.writes)
        self.assertIn((12, ADDR_GOAL_POSITION, 2048), backend.writes)
        self.assertIn((18, ADDR_GOAL_POSITION, 1652), backend.writes)

    def test_jog_bending_uses_safety_gate(self):
        backend = FakeBackend()
        agent = DxlAgent(default_robot_config(), backend)
        agent.arm()

        agent.jog_bending(25.0)

        self.assertIn((18, ADDR_GOAL_POSITION, 1936), backend.writes)


if __name__ == "__main__":
    unittest.main()
