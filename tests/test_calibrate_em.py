import csv
import tempfile
import time
import unittest
from pathlib import Path

from continuum_control.calibrate_em import (
    CalibrationError,
    CalibrationSettings,
    build_grid,
    csv_fieldnames,
    run_calibration,
    summarize_positions,
)
from continuum_control.config import load_em_config, load_robot_config
from continuum_control.dxl_agent import (
    ADDR_GOAL_POSITION,
    ADDR_HARDWARE_ERROR_STATUS,
    ADDR_OPERATING_MODE,
    ADDR_PRESENT_POSITION,
    ADDR_TORQUE_ENABLE,
    XL430_W250_MODEL_NUMBER,
)
from continuum_control.em_core import EMSample


def tf(x=0.0, y=0.0, z=0.0):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


class FakeBackend:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.writes = []
        self.positions = {12: 2048, 18: 1652}
        self.modes = {12: 3, 18: 3}

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def ping(self, dxl_id):
        return XL430_W250_MODEL_NUMBER if dxl_id in (12, 18) else None

    def read1(self, dxl_id, address):
        if address == ADDR_OPERATING_MODE:
            return self.modes[dxl_id]
        if address == ADDR_TORQUE_ENABLE:
            return 1
        if address == ADDR_HARDWARE_ERROR_STATUS:
            return 0
        raise AssertionError((dxl_id, address))

    def read2(self, dxl_id, address):
        raise AssertionError((dxl_id, address))

    def read4(self, dxl_id, address):
        if address == ADDR_PRESENT_POSITION:
            return self.positions[dxl_id]
        if address == ADDR_GOAL_POSITION:
            return self.positions[dxl_id]
        raise AssertionError((dxl_id, address))

    def write1(self, dxl_id, address, value):
        self.writes.append((dxl_id, address, value))

    def write2(self, dxl_id, address, value):
        self.writes.append((dxl_id, address, value))

    def write4(self, dxl_id, address, value):
        self.writes.append((dxl_id, address, value))
        if address == ADDR_GOAL_POSITION:
            self.positions[dxl_id] = value


class FakeEM:
    def __init__(self, valid=True):
        self.valid = valid
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def read_samples(self):
        now = time.monotonic()
        if not self.valid:
            return {
                "tip": EMSample("tip", "tip", 1, now, False, None, error="invalid transform"),
                "base": EMSample("base", "base", 0, now, True, tf()),
            }
        return {
            "tip": EMSample("tip", "tip", 1, now, True, tf(2.0, 3.0, 4.0)),
            "base": EMSample("base", "base", 0, now, True, tf(1.0, 1.0, 1.0)),
        }


class CalibrationTests(unittest.TestCase):
    def test_build_grid_validates_bounds(self):
        grid = build_grid(rotation_offsets=(-100, 0, 100), bending_degs=(-10, 0, 10))

        self.assertEqual(len(grid), 9)
        self.assertEqual(grid[0].rotation_offset_tick, -100)
        self.assertEqual(grid[0].bending_deg, -10.0)

        with self.assertRaisesRegex(CalibrationError, "rotation_offset_tick"):
            build_grid(rotation_offsets=(-101,), bending_degs=(0,))
        with self.assertRaisesRegex(CalibrationError, "bending_deg"):
            build_grid(rotation_offsets=(0,), bending_degs=(10.1,))

    def test_summarize_positions(self):
        stats = summarize_positions([(1.0, 2.0, 3.0), (3.0, 4.0, 7.0)])

        self.assertEqual(stats.mean, (2.0, 3.0, 5.0))
        self.assertEqual(stats.std, (1.0, 1.0, 2.0))
        self.assertEqual(stats.valid_count, 2)

    def test_csv_fieldnames_include_abort_reason(self):
        self.assertIn("abort_reason", csv_fieldnames())
        self.assertIn("rotation_offset_tick", csv_fieldnames())
        self.assertIn("bending_deg", csv_fieldnames())

    def test_run_calibration_commands_only_rotation_and_bending(self):
        robot_config = load_robot_config("config/robot.yaml")
        em_config = load_em_config("config/em.yaml")
        backend = FakeBackend()
        output = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)

        run_calibration(
            robot_config=robot_config,
            em_config=em_config,
            motor_backend=backend,
            em_agent=FakeEM(),
            output_path=output,
            settings=CalibrationSettings(
                rotation_offsets=(0,),
                bending_degs=(0.0,),
                settle_s=0.0,
                sample_s=0.05,
                min_valid=1,
            ),
            sleep=lambda _: None,
        )

        commanded_ids = {dxl_id for dxl_id, address, _ in backend.writes if address == ADDR_GOAL_POSITION}
        self.assertEqual(commanded_ids, {12, 18})
        with output.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["abort_reason"], "")
        self.assertEqual(rows[0]["valid_count"], "2")

    def test_run_calibration_records_invalid_em_abort(self):
        robot_config = load_robot_config("config/robot.yaml")
        em_config = load_em_config("config/em.yaml")
        output = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)

        run_calibration(
            robot_config=robot_config,
            em_config=em_config,
            motor_backend=FakeBackend(),
            em_agent=FakeEM(valid=False),
            output_path=output,
            settings=CalibrationSettings(
                rotation_offsets=(0,),
                bending_degs=(0.0,),
                settle_s=0.0,
                sample_s=0.05,
                min_valid=1,
            ),
            sleep=lambda _: None,
        )

        with output.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertIn("not enough valid EM samples", rows[0]["abort_reason"])

    def test_cleanup_returns_home_without_torque_off(self):
        robot_config = load_robot_config("config/robot.yaml")
        em_config = load_em_config("config/em.yaml")
        backend = FakeBackend()

        run_calibration(
            robot_config=robot_config,
            em_config=em_config,
            motor_backend=backend,
            em_agent=FakeEM(),
            output_path=Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name),
            settings=CalibrationSettings(
                rotation_offsets=(0,),
                bending_degs=(0.0,),
                settle_s=0.0,
                sample_s=0.05,
                min_valid=1,
            ),
            sleep=lambda _: None,
        )

        torque_off_writes = [
            write for write in backend.writes if write == (18, ADDR_TORQUE_ENABLE, 0)
        ]
        goal_writes = [write for write in backend.writes if write[1] == ADDR_GOAL_POSITION]
        self.assertEqual(torque_off_writes, [])
        self.assertIn((12, ADDR_GOAL_POSITION, 2048), goal_writes)
        self.assertIn((18, ADDR_GOAL_POSITION, 1652), goal_writes)

    def test_validation_failure_does_not_command_motors(self):
        robot_config = load_robot_config("config/robot.yaml")
        em_config = load_em_config("config/em.yaml")
        backend = FakeBackend()
        backend.modes[18] = 4

        with self.assertRaisesRegex(CalibrationError, "ID018 mode"):
            run_calibration(
                robot_config=robot_config,
                em_config=em_config,
                motor_backend=backend,
                em_agent=FakeEM(),
                output_path=Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name),
                settings=CalibrationSettings(
                    rotation_offsets=(0,),
                    bending_degs=(0.0,),
                    settle_s=0.0,
                    sample_s=0.05,
                    min_valid=1,
                ),
                sleep=lambda _: None,
            )

        goal_writes = [write for write in backend.writes if write[1] == ADDR_GOAL_POSITION]
        self.assertEqual(goal_writes, [])


if __name__ == "__main__":
    unittest.main()
