import tempfile
import textwrap
import unittest
from pathlib import Path

from continuum_control.config import ConfigError, load_em_config


class EMConfigTests(unittest.TestCase):
    def test_load_default_em_config(self):
        cfg = load_em_config("config/em.yaml")

        self.assertEqual(cfg.serial_port, "auto")
        self.assertEqual(len(cfg.sensors), 2)
        self.assertEqual(cfg.sensor_by_role("tip").tool_index, 10)
        self.assertEqual(cfg.sensor_by_role("base").tool_index, 11)

    def test_allows_aux_as_third_sensor(self):
        path = self._write(
            """
            serial_port: auto
            sensors:
              - {name: tip, role: tip, tool_index: 10}
              - {name: base, role: base, tool_index: 11}
              - {name: shape, role: aux, tool_index: 12}
            """
        )

        cfg = load_em_config(path)

        self.assertEqual(cfg.sensor_by_role("aux").name, "shape")

    def test_rejects_more_than_three_sensors(self):
        path = self._write(
            """
            serial_port: auto
            sensors:
              - {name: s1, role: tip, tool_index: 0}
              - {name: s2, role: base, tool_index: 1}
              - {name: s3, role: aux, tool_index: 2}
              - {name: s4, role: aux2, tool_index: 3}
            """
        )

        with self.assertRaisesRegex(ConfigError, "at most 3"):
            load_em_config(path)

    def test_rejects_duplicate_roles_and_tool_indices(self):
        path = self._write(
            """
            serial_port: auto
            sensors:
              - {name: tip1, role: tip, tool_index: 10}
              - {name: tip2, role: tip, tool_index: 10}
            """
        )

        with self.assertRaisesRegex(ConfigError, "unique"):
            load_em_config(path)

    def test_rejects_unknown_sensor_role(self):
        path = self._write(
            """
            serial_port: auto
            sensors:
              - {name: typo, role: tpi, tool_index: 10}
            """
        )

        with self.assertRaisesRegex(ConfigError, "role"):
            load_em_config(path)

    def test_rejects_non_finite_timing(self):
        path = self._write(
            """
            serial_port: auto
            timeout_s: .nan
            sensors:
              - {name: tip, role: tip, tool_index: 10}
            """
        )

        with self.assertRaisesRegex(ConfigError, "timeout_s"):
            load_em_config(path)

    def _write(self, body):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(textwrap.dedent(body))
        tmp.close()
        return Path(tmp.name)


if __name__ == "__main__":
    unittest.main()
