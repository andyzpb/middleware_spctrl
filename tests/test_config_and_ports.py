import unittest

from continuum_control.config import ConfigError, load_robot_config
from continuum_control.serial_ports import SerialPortError, resolve_serial_port


class ConfigAndPortTests(unittest.TestCase):
    def test_load_robot_config(self):
        cfg = load_robot_config("config/robot.yaml")

        self.assertEqual(cfg.baudrate, 1_000_000)
        self.assertEqual(cfg.axes["translation"].dxl_id, 8)
        self.assertEqual(cfg.axes["rotation"].dxl_id, 12)
        self.assertEqual(cfg.bending.dxl_id, 18)
        self.assertEqual(cfg.bending.required_mode, 3)
        self.assertEqual(cfg.bending.home_tick, 1652)
        self.assertEqual(cfg.bending.min_tick, 1368)
        self.assertEqual(cfg.bending.max_tick, 1936)

    def test_missing_config_fails(self):
        with self.assertRaises(ConfigError):
            load_robot_config("config/missing.yaml")

    def test_explicit_serial_port_is_used_directly(self):
        self.assertEqual(resolve_serial_port("/dev/cu.usbserial-FT123"), "/dev/cu.usbserial-FT123")
        self.assertEqual(resolve_serial_port("COM7"), "COM7")

    def test_auto_serial_port_filters_by_platform(self):
        self.assertEqual(
            resolve_serial_port(
                "auto",
                platform="darwin",
                candidates=["/dev/cu.usbserial-FT123", "/dev/tty.Bluetooth-Incoming-Port"],
            ),
            "/dev/cu.usbserial-FT123",
        )
        self.assertEqual(
            resolve_serial_port("auto", platform="linux", candidates=["/dev/ttyUSB0"]),
            "/dev/ttyUSB0",
        )
        self.assertEqual(
            resolve_serial_port("auto", platform="win32", candidates=["COM7"]),
            "COM7",
        )

    def test_auto_serial_port_fails_on_ambiguity(self):
        with self.assertRaises(SerialPortError):
            resolve_serial_port(
                "auto",
                platform="linux",
                candidates=["/dev/ttyUSB0", "/dev/ttyUSB1"],
            )

    def test_auto_serial_port_fails_when_none_found(self):
        with self.assertRaises(SerialPortError):
            resolve_serial_port("auto", platform="darwin", candidates=["/dev/tty.not-u2d2"])


if __name__ == "__main__":
    unittest.main()
