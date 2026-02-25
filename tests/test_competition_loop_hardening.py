import unittest
from unittest.mock import patch

from config.settings import Settings
try:
    import numpy as np
except ImportError:  # pragma: no cover - environment-dependent
    np = None
try:
    from src.network import FrameFetchResult, FrameFetchStatus, SendResultStatus
    import main as main_module
except Exception:  # pragma: no cover - environment-dependent
    FrameFetchResult = None
    FrameFetchStatus = None
    SendResultStatus = None
    main_module = None


class _DummyDetector:
    detect_calls = 0

    def detect(self, frame):
        _DummyDetector.detect_calls += 1
        return []


class _DummyMovement:
    def annotate(self, detections, frame=None):
        return detections


class _DummyOdometry:
    def update(self, frame, frame_data):
        return {"x": 0.0, "y": 0.0, "z": 0.0}


class _FakeNetwork:
    frame_results = []
    timeout_snapshots = []
    download_calls = 0
    send_calls = 0

    def __init__(self, base_url=None, simulation_mode=None):
        pass

    @classmethod
    def reset(cls):
        cls.frame_results = []
        cls.timeout_snapshots = []
        cls.download_calls = 0
        cls.send_calls = 0
        _DummyDetector.detect_calls = 0

    def start_session(self):
        return True

    def get_frame(self):
        return self.frame_results.pop(0)

    def consume_timeout_counters(self):
        if self.timeout_snapshots:
            return self.timeout_snapshots.pop(0)
        return {"fetch": 0, "image": 0, "submit": 0}

    def download_image(self, frame_data):
        _FakeNetwork.download_calls += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def send_result(
        self,
        frame_id,
        detected_objects,
        detected_translation,
        frame_data=None,
        frame_shape=None,
        degrade=False,
    ):
        _FakeNetwork.send_calls += 1
        return SendResultStatus.ACKED


@unittest.skipUnless(
    np is not None
    and FrameFetchResult is not None
    and FrameFetchStatus is not None
    and SendResultStatus is not None
    and main_module is not None,
    "runtime deps are missing",
)
class TestCompetitionLoopHardening(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "DEBUG": Settings.DEBUG,
            "MAX_FRAMES": Settings.MAX_FRAMES,
            "LOOP_DELAY": Settings.LOOP_DELAY,
            "FPS_REPORT_INTERVAL": Settings.FPS_REPORT_INTERVAL,
            "DEGRADE_FETCH_ONLY_ENABLED": Settings.DEGRADE_FETCH_ONLY_ENABLED,
        }
        Settings.DEBUG = False
        Settings.MAX_FRAMES = 50
        Settings.LOOP_DELAY = 0.0
        Settings.FPS_REPORT_INTERVAL = 99999
        Settings.DEGRADE_FETCH_ONLY_ENABLED = False
        _FakeNetwork.reset()
        self.summary_calls = []

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    def test_duplicate_frame_dropped_before_processing(self):
        _FakeNetwork.frame_results = [
            FrameFetchResult(
                status=FrameFetchStatus.OK,
                frame_data={"frame_id": "f1", "frame_url": "/f1.jpg", "gps_health": 1},
                is_duplicate=False,
            ),
            FrameFetchResult(
                status=FrameFetchStatus.OK,
                frame_data={"frame_id": "f1", "frame_url": "/f1.jpg", "gps_health": 1},
                is_duplicate=True,
            ),
            FrameFetchResult(status=FrameFetchStatus.END_OF_STREAM),
        ]
        _FakeNetwork.timeout_snapshots = [{"fetch": 0, "image": 0, "submit": 0}] * 8

        with patch("src.network.NetworkManager", _FakeNetwork), \
             patch.object(main_module, "ObjectDetector", _DummyDetector), \
             patch.object(main_module, "MovementEstimator", _DummyMovement), \
             patch.object(main_module, "VisualOdometry", _DummyOdometry), \
             patch.object(main_module, "_print_summary", side_effect=lambda *a, **kw: self.summary_calls.append(kw)):
            main_module.run_competition(main_module.Logger("Test"))

        self.assertEqual(_FakeNetwork.download_calls, 1)
        self.assertEqual(_FakeNetwork.send_calls, 1)
        self.assertEqual(_DummyDetector.detect_calls, 1)
        kpi = self.summary_calls[-1]["kpi_counters"]
        self.assertEqual(kpi["frame_duplicate_drop"], 1)

    def test_transient_fetch_timeout_recovers(self):
        _FakeNetwork.frame_results = [
            FrameFetchResult(status=FrameFetchStatus.TRANSIENT_ERROR, error_type="retries_exhausted"),
            FrameFetchResult(
                status=FrameFetchStatus.OK,
                frame_data={"frame_id": "f2", "frame_url": "/f2.jpg", "gps_health": 1},
                is_duplicate=False,
            ),
            FrameFetchResult(status=FrameFetchStatus.END_OF_STREAM),
        ]
        _FakeNetwork.timeout_snapshots = [
            {"fetch": 1, "image": 0, "submit": 0},
            {"fetch": 0, "image": 0, "submit": 0},
            {"fetch": 0, "image": 0, "submit": 0},
            {"fetch": 0, "image": 0, "submit": 0},
            {"fetch": 0, "image": 0, "submit": 0},
        ]

        with patch("src.network.NetworkManager", _FakeNetwork), \
             patch.object(main_module, "ObjectDetector", _DummyDetector), \
             patch.object(main_module, "MovementEstimator", _DummyMovement), \
             patch.object(main_module, "VisualOdometry", _DummyOdometry), \
             patch.object(main_module, "_print_summary", side_effect=lambda *a, **kw: self.summary_calls.append(kw)):
            main_module.run_competition(main_module.Logger("Test"))

        self.assertEqual(_FakeNetwork.send_calls, 1)
        kpi = self.summary_calls[-1]["kpi_counters"]
        self.assertEqual(kpi["timeout_fetch"], 1)


if __name__ == "__main__":
    unittest.main()
