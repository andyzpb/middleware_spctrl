import io
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout

from continuum_control.config import EMConfig, SensorConfig
from continuum_control.em_cli import build_parser, main
from continuum_control.em_core import EMSample


def tf(x=0.0, y=0.0, z=0.0):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


class FakeAgent:
    calls = []

    def __init__(self, config):
        self.config = config

    def start(self):
        self.calls.append(("start",))

    def close(self):
        self.calls.append(("close",))

    def read_samples(self):
        self.calls.append(("read_samples",))
        return {
            "tip": EMSample("tip", "tip", 10, 10.0, True, tf(2.0, 3.0, 4.0)),
            "base": EMSample("base", "base", 11, 10.0, True, tf(1.0, 1.0, 1.0)),
        }

    def scan_indices(self, max_index):
        self.calls.append(("scan_indices", max_index))
        return [
            EMSample("scan", "index_0", 0, 10.0, True, tf(4.0, 5.0, 6.0)),
            EMSample("scan", "index_1", 1, 10.0, False, None, error="invalid transform"),
        ]


class EMCliTests(unittest.TestCase):
    def setUp(self):
        FakeAgent.calls = []

    def test_parser_exposes_em_debug_commands(self):
        commands = build_parser()._subparsers._group_actions[0].choices

        self.assertEqual(set(commands), {"status", "read", "pair", "scan"})

    def test_status_prints_config_without_opening_tracker(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = main(["--config", "config/em.yaml", "status"], agent_factory=FakeAgent)

        self.assertEqual(code, 0)
        self.assertEqual(FakeAgent.calls, [])
        self.assertIn("role=tip", stdout.getvalue())

    def test_read_starts_agent_and_prints_samples(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = main(["--config", "config/em.yaml", "read", "--samples", "1"], agent_factory=FakeAgent)

        self.assertEqual(code, 0)
        self.assertEqual(FakeAgent.calls, [("start",), ("read_samples",), ("close",)])
        self.assertIn("role=tip", stdout.getvalue())
        self.assertIn("x_mm=2.000", stdout.getvalue())

    def test_pair_requires_tip_and_base_roles(self):
        class TipOnlyAgent(FakeAgent):
            def read_samples(self):
                return {"tip": EMSample("tip", "tip", 10, 10.0, True, tf())}

        cfg = EMConfig(
            serial_port="auto",
            ports_to_probe=20,
            timeout_s=0.5,
            sample_hz=40.0,
            sensors=(SensorConfig("tip", "tip", 10),),
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            code = main(["pair", "--samples", "1"], config_loader=lambda _: cfg, agent_factory=TipOnlyAgent)

        self.assertEqual(code, 2)
        self.assertIn("missing base", stderr.getvalue())

    def test_pair_continues_after_transient_invalid_sample(self):
        class WarmupAgent(FakeAgent):
            def __init__(self, config):
                super().__init__(config)
                self.count = 0

            def read_samples(self):
                self.calls.append(("read_samples",))
                self.count += 1
                now = time.monotonic()
                if self.count == 1:
                    return {
                        "tip": EMSample("tip", "tip", 1, now, False, None, error="invalid transform"),
                        "base": EMSample("base", "base", 0, now, True, tf(1.0, 1.0, 1.0)),
                    }
                return {
                    "tip": EMSample("tip", "tip", 1, now, True, tf(2.0, 3.0, 4.0)),
                    "base": EMSample("base", "base", 0, now, True, tf(1.0, 1.0, 1.0)),
                }

        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = main(["--config", "config/em.yaml", "pair", "--samples", "2"], agent_factory=WarmupAgent)

        self.assertEqual(code, 0)
        self.assertIn("pair invalid: tip invalid: invalid transform", stdout.getvalue())
        self.assertIn("tip_in_base x_mm=1.000 y_mm=2.000 z_mm=3.000", stdout.getvalue())

    def test_scan_prints_raw_indices(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = main(["--config", "config/em.yaml", "scan", "--max-index", "1"], agent_factory=FakeAgent)

        self.assertEqual(code, 0)
        self.assertEqual(FakeAgent.calls, [("start",), ("scan_indices", 1), ("close",)])
        self.assertIn("index=0 valid=1 x_mm=4.000", stdout.getvalue())
        self.assertIn("index=1 valid=0 error=invalid transform", stdout.getvalue())

    def test_scan_rejects_negative_max_index_before_opening_tracker(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            code = main(["--config", "config/em.yaml", "scan", "--max-index", "-1"], agent_factory=FakeAgent)

        self.assertEqual(code, 2)
        self.assertEqual(FakeAgent.calls, [])
        self.assertIn("max-index must be >= 0", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
