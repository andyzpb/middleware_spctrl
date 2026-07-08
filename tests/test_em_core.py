import unittest

from continuum_control.em_core import EMError, EMSample, fresh_samples, relative_transform


def tf(x=0.0, y=0.0, z=0.0, r=None):
    r = r or ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return (
        (r[0][0], r[0][1], r[0][2], x),
        (r[1][0], r[1][1], r[1][2], y),
        (r[2][0], r[2][1], r[2][2], z),
        (0.0, 0.0, 0.0, 1.0),
    )


class EMCoreTests(unittest.TestCase):
    def test_relative_transform_reports_tip_in_base_frame(self):
        base = EMSample("base", "base", 11, 10.0, True, tf(10.0, 0.0, 0.0))
        tip = EMSample("tip", "tip", 10, 10.0, True, tf(12.0, 3.0, 4.0))

        rel = relative_transform(base, tip)

        self.assertEqual(rel[0][3], 2.0)
        self.assertEqual(rel[1][3], 3.0)
        self.assertEqual(rel[2][3], 4.0)

    def test_relative_transform_uses_base_orientation(self):
        rot_z_90 = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        base = EMSample("base", "base", 11, 10.0, True, tf(0.0, 0.0, 0.0, rot_z_90))
        tip = EMSample("tip", "tip", 10, 10.0, True, tf(0.0, 1.0, 0.0))

        rel = relative_transform(base, tip)

        self.assertEqual(rel[0][3], 1.0)
        self.assertEqual(rel[1][3], 0.0)
        self.assertEqual(rel[2][3], 0.0)

    def test_fresh_samples_rejects_stale_or_invalid_roles(self):
        samples = {
            "tip": EMSample("tip", "tip", 10, 1.0, True, tf()),
            "base": EMSample("base", "base", 11, 0.0, False, None, error="missing transform"),
        }

        with self.assertRaisesRegex(EMError, "base invalid"):
            fresh_samples(samples, ("tip", "base"), now_s=1.1, timeout_s=0.5)

        samples["base"] = EMSample("base", "base", 11, 0.0, True, tf())
        with self.assertRaisesRegex(EMError, "base stale"):
            fresh_samples(samples, ("tip", "base"), now_s=1.1, timeout_s=0.5)

    def test_sample_rejects_non_finite_transform(self):
        with self.assertRaisesRegex(EMError, "transform"):
            EMSample("tip", "tip", 10, 1.0, True, tf(float("nan")))


if __name__ == "__main__":
    unittest.main()
