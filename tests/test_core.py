import unittest

from continuum_control.core import (
    Command,
    SafetyError,
    SafetyGate,
    bending_deg_to_tick,
    bending_tick_to_deg,
    default_robot_config,
)


class CoreTests(unittest.TestCase):
    def test_bending_map_calibration_points(self):
        cfg = default_robot_config()

        self.assertEqual(bending_deg_to_tick(-25.0, cfg), 1368)
        self.assertEqual(bending_deg_to_tick(0.0, cfg), 1652)
        self.assertEqual(bending_deg_to_tick(25.0, cfg), 1936)

        self.assertAlmostEqual(bending_tick_to_deg(1368, cfg), -25.0)
        self.assertAlmostEqual(bending_tick_to_deg(1652, cfg), 0.0)
        self.assertAlmostEqual(bending_tick_to_deg(1936, cfg), 25.0)

    def test_external_bending_command_outside_range_is_rejected(self):
        gate = SafetyGate(default_robot_config())

        with self.assertRaises(SafetyError):
            gate.command_to_goals(Command(bending_deg=25.1, source="operator"))

    def test_controller_bending_command_is_clamped_and_marked_saturated(self):
        gate = SafetyGate(default_robot_config())

        goal = gate.command_to_goals(Command(bending_deg=30.0, source="controller"))

        self.assertEqual(goal.ticks, {18: 1936})
        self.assertTrue(goal.saturated)

    def test_home_goal_contains_all_three_ids(self):
        gate = SafetyGate(default_robot_config())

        self.assertEqual(gate.home_goal().ticks, {8: 2048, 12: 2048, 18: 1652})


if __name__ == "__main__":
    unittest.main()
