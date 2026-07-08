import unittest

from continuum_control.aurora_agent import AuroraAgent
from continuum_control.config import EMConfig, SensorConfig


def tf(x=0.0, y=0.0, z=0.0):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


class FakeTracker:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.closed = False

    def start_tracking(self):
        self.started = True

    def stop_tracking(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def get_frame(self):
        tracking = [None] * 13
        tracking[10] = tf(1.0, 2.0, 3.0)
        tracking[11] = tf(4.0, 5.0, 6.0)
        return None, [0.0] * 13, None, tracking, [None] * 13


class AuroraAgentTests(unittest.TestCase):
    def test_reads_configured_role_samples_without_importing_hardware_sdk(self):
        cfg = EMConfig(
            serial_port="auto",
            ports_to_probe=20,
            timeout_s=0.5,
            sample_hz=40.0,
            sensors=(
                SensorConfig("tip", "tip", 10),
                SensorConfig("base", "base", 11),
            ),
        )
        tracker = FakeTracker()
        agent = AuroraAgent(cfg, tracker=tracker, clock=lambda: 100.0)

        agent.start()
        samples = agent.read_samples()
        agent.close()

        self.assertTrue(tracker.started)
        self.assertTrue(tracker.stopped)
        self.assertTrue(tracker.closed)
        self.assertEqual(samples["tip"].timestamp_s, 100.0)
        self.assertEqual(samples["tip"].position_mm, (1.0, 2.0, 3.0))
        self.assertEqual(samples["base"].position_mm, (4.0, 5.0, 6.0))

    def test_invalid_transform_returns_invalid_sample(self):
        cfg = EMConfig(
            serial_port="auto",
            ports_to_probe=20,
            timeout_s=0.5,
            sample_hz=40.0,
            sensors=(SensorConfig("tip", "tip", 10),),
        )
        tracker = FakeTracker()
        tracker.get_frame = lambda: (None, [0.0] * 11, None, [None] * 11, [None] * 11)

        sample = AuroraAgent(cfg, tracker=tracker, clock=lambda: 100.0).read_samples()["tip"]

        self.assertFalse(sample.valid)
        self.assertIn("invalid transform", sample.error)

    def test_close_before_start_does_not_stop_tracking(self):
        cfg = EMConfig(
            serial_port="auto",
            ports_to_probe=20,
            timeout_s=0.5,
            sample_hz=40.0,
            sensors=(SensorConfig("tip", "tip", 10),),
        )
        tracker = FakeTracker()

        AuroraAgent(cfg, tracker=tracker).close()

        self.assertFalse(tracker.stopped)
        self.assertTrue(tracker.closed)

    def test_scan_indices_reads_raw_tracking_slots(self):
        cfg = EMConfig(
            serial_port="auto",
            ports_to_probe=20,
            timeout_s=0.5,
            sample_hz=40.0,
            sensors=(SensorConfig("tip", "tip", 10),),
        )
        tracker = FakeTracker()

        rows = AuroraAgent(cfg, tracker=tracker, clock=lambda: 100.0).scan_indices(10)

        self.assertEqual(rows[0].tool_index, 0)
        self.assertFalse(rows[0].valid)
        self.assertEqual(rows[1].tool_index, 1)
        self.assertFalse(rows[1].valid)
        self.assertEqual(rows[10].position_mm, (1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
