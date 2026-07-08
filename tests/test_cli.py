import io
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout

from continuum_control.cli import build_parser, main


class FakeAgent:
    calls = []

    def __init__(self, config):
        self.config = config

    def open(self):
        self.calls.append(("open", self.config.serial_port))

    def close(self):
        self.calls.append(("close",))

    def status(self):
        self.calls.append(("status",))
        return []

    def arm(self):
        self.calls.append(("arm", self.config.serial_port))

    def disarm(self):
        self.calls.append(("disarm",))

    def home(self):
        self.calls.append(("home",))
        return None

    def jog_bending(self, deg):
        self.calls.append(("jog_bending", deg))
        return None


class CliTests(unittest.TestCase):
    def setUp(self):
        FakeAgent.calls = []

    def test_parser_exposes_same_cross_platform_commands(self):
        parser = build_parser()
        commands = parser._subparsers._group_actions[0].choices

        self.assertEqual(set(commands), {"status", "arm", "disarm", "home", "jog"})

    def test_parser_accepts_jog_bending_degrees(self):
        args = build_parser().parse_args(["--port", "COM7", "jog", "--bending-deg", "5"])

        self.assertEqual(args.command, "jog")
        self.assertEqual(args.port, "COM7")
        self.assertEqual(args.bending_deg, 5.0)

    def test_main_jog_arms_then_jogs_without_touching_real_hardware(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = main(
                ["--config", "config/robot.yaml", "--port", "COM7", "jog", "--bending-deg", "5"],
                agent_factory=FakeAgent,
            )

        self.assertEqual(code, 0)
        self.assertEqual(FakeAgent.calls, [("arm", "COM7"), ("jog_bending", 5.0), ("close",)])

    def test_main_status_opens_reads_and_closes(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = main(
                ["--config", "config/robot.yaml", "--port", "/dev/ttyUSB0", "status"],
                agent_factory=FakeAgent,
            )

        self.assertEqual(code, 0)
        self.assertEqual(FakeAgent.calls, [("open", "/dev/ttyUSB0"), ("status",), ("close",)])

    def test_main_rejects_unsafe_jog_before_constructing_agent(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            code = main(
                ["--config", "config/robot.yaml", "--port", "COM7", "jog", "--bending-deg", "25.1"],
                agent_factory=FakeAgent,
            )

        self.assertEqual(code, 2)
        self.assertEqual(FakeAgent.calls, [])
        self.assertIn("bending_deg 25.1 outside", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
