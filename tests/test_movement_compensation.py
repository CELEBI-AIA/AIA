import unittest

try:
    import cv2
except ImportError:  # pragma: no cover - environment-dependent
    cv2 = None
import numpy as np

from config.settings import Settings
try:
    from src.movement import MovementEstimator
except Exception:  # pragma: no cover - environment-dependent
    MovementEstimator = None


@unittest.skipUnless(cv2 is not None and MovementEstimator is not None, "opencv/runtime deps are missing")
class TestMovementCompensation(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MOTION_COMP_ENABLED": Settings.MOTION_COMP_ENABLED,
            "MOVEMENT_MIN_HISTORY": Settings.MOVEMENT_MIN_HISTORY,
            "MOVEMENT_THRESHOLD_PX": Settings.MOVEMENT_THRESHOLD_PX,
            "MOVEMENT_MATCH_DISTANCE_PX": Settings.MOVEMENT_MATCH_DISTANCE_PX,
            "MOTION_COMP_MIN_FEATURES": Settings.MOTION_COMP_MIN_FEATURES,
        }
        Settings.MOVEMENT_MIN_HISTORY = 2
        Settings.MOVEMENT_THRESHOLD_PX = 8.0
        Settings.MOVEMENT_MATCH_DISTANCE_PX = 200.0
        Settings.MOTION_COMP_MIN_FEATURES = 20

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    def _frame(self, shift_x: int = 0) -> np.ndarray:
        rng = np.random.default_rng(42)
        base = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(600):
            x = int(rng.integers(5, 315))
            y = int(rng.integers(5, 315))
            cv2.circle(base, (x, y), 1, (255, 255, 255), -1)
        m = np.float32([[1, 0, shift_x], [0, 1, 0]])
        return cv2.warpAffine(base, m, (320, 320))

    def _vehicle(self, x1: int, y1: int, x2: int, y2: int):
        return [{
            "cls": "0",
            "top_left_x": x1,
            "top_left_y": y1,
            "bottom_right_x": x2,
            "bottom_right_y": y2,
            "landing_status": "-1",
        }]

    def test_stationary_vehicle_with_camera_pan_marked_static(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()

        f1 = self._frame(0)
        f2 = self._frame(12)

        est.annotate(self._vehicle(100, 100, 140, 140), frame=f1)
        out = est.annotate(self._vehicle(112, 100, 152, 140), frame=f2)

        self.assertEqual(out[0]["motion_status"], "0")

    def test_actual_motion_preserved_with_compensation(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()

        f1 = self._frame(0)
        f2 = self._frame(12)

        est.annotate(self._vehicle(100, 100, 140, 140), frame=f1)
        out = est.annotate(self._vehicle(124, 100, 164, 140), frame=f2)

        self.assertEqual(out[0]["motion_status"], "1")

    def test_low_feature_fallback_no_crash(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()

        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        est.annotate(self._vehicle(100, 100, 140, 140), frame=blank)
        out = est.annotate(self._vehicle(108, 100, 148, 140), frame=blank)

        self.assertIn(out[0]["motion_status"], {"0", "1", "-1"})

    def test_first_frame_warmup_not_moving(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()

        out = est.annotate(self._vehicle(100, 100, 140, 140), frame=self._frame(0))
        self.assertEqual(out[0]["motion_status"], "0")


if __name__ == "__main__":
    unittest.main()
