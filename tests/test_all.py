"""
AIA — Tüm Birim Testleri (Konsolide)
=====================================
Bu dosya tüm test modüllerini tek bir yerde toplar.
Çalıştırma: python -m pytest tests/test_all.py -v
"""

import copy
import json
import os
import sys
import time
import types
import unittest
from collections import deque
from unittest.mock import Mock, MagicMock, patch, mock_open

import numpy as np
import pytest

from config.settings import Settings

# ─── Soft imports (environment-dependent) ────────────────────────────────────
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from src.network import (
        NetworkManager,
        FrameFetchResult,
        FrameFetchStatus,
        SendResultStatus,
    )
except Exception:  # pragma: no cover
    NetworkManager = None
    FrameFetchResult = None
    FrameFetchStatus = None
    SendResultStatus = None

try:
    from src.detection import ObjectDetector
except Exception:  # pragma: no cover
    ObjectDetector = None

try:
    from src.movement import MovementEstimator
except Exception:  # pragma: no cover
    MovementEstimator = None

try:
    import main as main_module
except Exception:  # pragma: no cover
    main_module = None

from src.send_state import apply_send_result_status
from src.resilience import ResilienceState, SessionResilienceController
from src.runtime_profile import apply_runtime_profile
from src.utils import Logger, log_json_to_disk, _sanitize_log_component, _prune_old_logs
from main import run_simulation


# =============================================================================
#  §1  UTILS TESTS
# =============================================================================

class TestLogger:
    @patch('builtins.print')
    def test_logger_info(self, mock_print):
        Logger("TestModule").info("Test message")
        assert mock_print.called

    @patch('builtins.print')
    def test_logger_debug(self, mock_print):
        logger = Logger("TestModule")
        Settings.DEBUG = True
        logger.debug("show")
        assert mock_print.called
        mock_print.reset_mock()
        Settings.DEBUG = False
        logger.debug("hide")
        assert not mock_print.called
        Settings.DEBUG = True

    @patch('builtins.print')
    def test_logger_error(self, mock_print):
        Logger("TestModule").error("Error")
        assert mock_print.called

    @patch('builtins.print')
    def test_logger_warn(self, mock_print):
        Logger("TestModule").warn("Warn")
        assert mock_print.called

    @patch('builtins.print')
    def test_logger_success(self, mock_print):
        Logger("TestModule").success("OK")
        assert mock_print.called


class TestSanitizeAndLogs:
    def test_sanitize_log_component(self):
        assert _sanitize_log_component("valid_name") == "valid_name"
        assert _sanitize_log_component("invalid?name!") == "invalid_name_"
        assert _sanitize_log_component("") == "general"

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.utils._prune_old_logs')
    def test_log_json_to_disk(self, mock_prune, mock_file, mock_makedirs):
        data = {"key": "value"}
        Settings.LOG_DIR = "/fake/dir"
        log_json_to_disk(data, direction="test_dir", tag="test_tag")
        mock_makedirs.assert_called_with("/fake/dir", exist_ok=True)
        mock_file.assert_called_once()
        mock_prune.assert_called_once_with("/fake/dir")
        written = "".join(c.args[0] for c in mock_file().write.call_args_list)
        assert json.loads(written) == data

    @patch('os.listdir')
    @patch('os.path.getmtime')
    @patch('os.remove')
    def test_prune_old_logs(self, mock_remove, mock_getmtime, mock_listdir):
        mock_listdir.return_value = ["log1.json", "log2.json", "log3.json"]
        mock_getmtime.side_effect = [1.0, 3.0, 2.0]
        Settings.LOG_MAX_FILES = 1
        _prune_old_logs("/fake/dir")
        assert mock_remove.call_count == 2


# =============================================================================
#  §2  RUNTIME PROFILE TESTS
# =============================================================================

class TestRuntimeProfile:
    def test_off(self):
        orig_tta = Settings.AUGMENTED_INFERENCE
        orig_fp16 = Settings.HALF_PRECISION
        apply_runtime_profile("off")
        assert Settings.AUGMENTED_INFERENCE == orig_tta
        assert Settings.HALF_PRECISION == orig_fp16

    def test_balanced(self):
        Settings.AUGMENTED_INFERENCE = True
        Settings.HALF_PRECISION = True
        apply_runtime_profile("balanced")
        assert Settings.AUGMENTED_INFERENCE is False
        assert Settings.HALF_PRECISION is True

    def test_max(self):
        Settings.AUGMENTED_INFERENCE = True
        Settings.HALF_PRECISION = True
        apply_runtime_profile("max")
        assert Settings.AUGMENTED_INFERENCE is False
        assert Settings.HALF_PRECISION is False

    def test_invalid(self):
        with pytest.raises(ValueError):
            apply_runtime_profile("invalid_profile")


# =============================================================================
#  §3  SEND STATE TESTS
# =============================================================================

class TestSendState:
    @staticmethod
    def _counters():
        return {"send_ok": 0, "send_fallback_ok": 0, "send_fail": 0, "send_permanent_reject": 0}

    def test_acked(self):
        c = self._counters()
        p, abort, ok = apply_send_result_status("acked", {"x": 1}, c)
        assert p is None and not abort and ok and c["send_ok"] == 1

    def test_fallback_acked(self):
        c = self._counters()

        class FS:
            value = "fallback_acked"

        p, abort, ok = apply_send_result_status(FS(), {"x": 1}, c)
        assert p is None and not abort and ok
        assert c["send_ok"] == 1 and c["send_fallback_ok"] == 1

    def test_permanent_rejected(self):
        c = self._counters()
        p, abort, ok = apply_send_result_status("permanent_rejected", {"x": 1}, c)
        assert p == {"x": 1} and abort and not ok
        assert c["send_fail"] == 1 and c["send_permanent_reject"] == 1

    def test_other_error(self):
        c = self._counters()
        p, abort, ok = apply_send_result_status("transient_error", {"x": 1}, c)
        assert p == {"x": 1} and not abort and not ok
        assert c["send_fail"] == 1 and c["send_permanent_reject"] == 0


# =============================================================================
#  §4  MAIN ACK STATE MACHINE TESTS
# =============================================================================

class TestMainAckStateMachine:
    @staticmethod
    def _counters():
        return {"send_ok": 0, "send_fail": 0, "send_fallback_ok": 0, "send_permanent_reject": 0}

    def test_retryable_failure_keeps_pending(self):
        pending = {"frame_id": "f-1"}
        c = self._counters()
        p, abort, ok = apply_send_result_status("retryable_failure", pending, c)
        assert p is pending and not abort and not ok and c["send_fail"] == 1

    def test_fallback_acked_clears_pending(self):
        c = self._counters()
        p, abort, ok = apply_send_result_status("fallback_acked", {"frame_id": "f-2"}, c)
        assert p is None and not abort and ok
        assert c["send_ok"] == 1 and c["send_fallback_ok"] == 1

    def test_permanent_rejected_aborts(self):
        c = self._counters()
        pending = {"frame_id": "f-3"}
        p, abort, ok = apply_send_result_status("permanent_rejected", pending, c)
        assert p is pending and abort and not ok
        assert c["send_fail"] == 1 and c["send_permanent_reject"] == 1


# =============================================================================
#  §5  SESSION RESILIENCE TESTS
# =============================================================================

class _StubLog:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warn(self, msg):
        self.lines.append(("warn", msg))

    def error(self, msg):
        self.lines.append(("error", msg))


class TestSessionResilience:
    def _ctrl(self):
        return SessionResilienceController(_StubLog())

    def _setup_settings(self):
        Settings.CB_TRANSIENT_WINDOW_SEC = 2.0
        Settings.CB_TRANSIENT_MAX_EVENTS = 3
        Settings.CB_OPEN_COOLDOWN_SEC = 0.2
        Settings.CB_MAX_OPEN_CYCLES = 2
        Settings.CB_SESSION_MAX_TRANSIENT_SEC = 0.4

    def test_transient_window_opens_breaker(self):
        self._setup_settings()
        c = self._ctrl()
        c.on_fetch_transient()
        c.on_fetch_transient()
        assert c.state == ResilienceState.DEGRADED
        c.on_fetch_transient()
        assert c.state == ResilienceState.OPEN
        assert c.stats.breaker_open_count == 1

    def test_open_to_half_open_to_normal_recovery(self):
        self._setup_settings()
        c = self._ctrl()
        c.on_fetch_transient()
        c.on_fetch_transient()
        c.on_fetch_transient()
        assert c.state == ResilienceState.OPEN
        assert not c.before_fetch()
        time.sleep(0.25)
        assert c.before_fetch()
        assert c.state == ResilienceState.DEGRADED
        c.on_success_cycle()
        assert c.state == ResilienceState.NORMAL
        assert c.stats.recovered_count >= 1

    def test_session_wall_clock_abort(self):
        self._setup_settings()
        c = self._ctrl()
        c.on_ack_failure()
        assert c.state == ResilienceState.DEGRADED
        time.sleep(0.45)
        reason = c.should_abort()
        assert reason is not None and "Transient wall time exceeded" in reason

    def test_breaker_open_cycles_abort(self):
        self._setup_settings()
        c = self._ctrl()
        c.stats.breaker_open_count = 3
        reason = c.should_abort()
        assert reason is not None and "Breaker open cycles exceeded" in reason


# =============================================================================
#  §6  MAIN / SIMULATION TESTS
# =============================================================================

@patch('src.data_loader.DatasetLoader')
@patch('main.ObjectDetector')
@patch('main.VisualOdometry')
@patch('main.MovementEstimator')
@patch('main.Logger')
@patch('main.Visualizer')
@patch('src.image_matcher.ImageMatcher')
def test_run_simulation_stops_on_max_frames(
    MockImageMatcher, MockVisualizer, MockLogger,
    MockEstimator, MockOdometry, MockDetector, MockDatasetLoader,
):
    mock_loader = MockDatasetLoader.return_value
    mock_loader.is_ready = True
    mock_loader.__iter__.return_value = [
        {"frame": None, "frame_idx": 0, "server_data": {}, "gps_health": 1},
        {"frame": None, "frame_idx": 1, "server_data": {}, "gps_health": 1},
    ]
    MockDetector.return_value.detect.return_value = []
    MockEstimator.return_value.annotate.return_value = []
    MockOdometry.return_value.update.return_value = {"x": 0.0, "y": 0.0, "z": 0.0}
    MockImageMatcher.return_value.match.return_value = []

    with patch('config.settings.Settings.MAX_FRAMES', 1):
        run_simulation(MockLogger(), prefer_vid=False, show=False, save=False)

    assert mock_loader.__iter__.called
    assert MockDetector.return_value.detect.call_count == 1


# =============================================================================
#  §7  RIDER SUPPRESSION TESTS
# =============================================================================

@unittest.skipUnless(ObjectDetector is not None, "detection deps missing")
class TestRiderSuppression(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "RIDER_SUPPRESS_ENABLED": Settings.RIDER_SUPPRESS_ENABLED,
            "RIDER_OVERLAP_THRESHOLD": Settings.RIDER_OVERLAP_THRESHOLD,
            "RIDER_IOU_THRESHOLD": Settings.RIDER_IOU_THRESHOLD,
            "RIDER_SOURCE_CLASSES": Settings.RIDER_SOURCE_CLASSES,
        }
        Settings.RIDER_SUPPRESS_ENABLED = True
        Settings.RIDER_OVERLAP_THRESHOLD = 0.35
        Settings.RIDER_IOU_THRESHOLD = 0.15
        Settings.RIDER_SOURCE_CLASSES = (1, 3, 10)

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    @staticmethod
    def _det(cls_int, source_cls_id, bbox):
        x1, y1, x2, y2 = bbox
        return {
            "cls_int": cls_int, "cls": str(cls_int),
            "source_cls_id": source_cls_id, "bbox": (x1, y1, x2, y2),
            "top_left_x": x1, "top_left_y": y1,
            "bottom_right_x": x2, "bottom_right_y": y2, "confidence": 0.9,
        }

    def test_person_over_bicycle_is_suppressed(self):
        person = self._det(1, 0, (100, 100, 140, 160))
        bicycle = self._det(0, 1, (95, 105, 145, 165))
        out = ObjectDetector._suppress_rider_persons([person, bicycle])
        self.assertEqual(sum(1 for d in out if d["cls_int"] == 1), 0)
        self.assertEqual(sum(1 for d in out if d["cls_int"] == 0), 1)

    def test_person_over_car_not_suppressed(self):
        person = self._det(1, 0, (100, 100, 140, 160))
        car = self._det(0, 2, (95, 105, 145, 165))
        out = ObjectDetector._suppress_rider_persons([person, car])
        self.assertEqual(sum(1 for d in out if d["cls_int"] == 1), 1)

    def test_low_overlap_not_suppressed(self):
        person = self._det(1, 0, (100, 100, 140, 160))
        bike = self._det(0, 1, (200, 200, 240, 260))
        out = ObjectDetector._suppress_rider_persons([person, bike])
        self.assertEqual(sum(1 for d in out if d["cls_int"] == 1), 1)

    def test_multi_person_only_overlapping_suppressed(self):
        rider = self._det(1, 0, (100, 100, 140, 160))
        walker = self._det(1, 0, (220, 120, 260, 180))
        moto = self._det(0, 3, (95, 105, 145, 165))
        out = ObjectDetector._suppress_rider_persons([rider, walker, moto])
        self.assertEqual(sum(1 for d in out if d["cls_int"] == 1), 1)
        self.assertEqual(sum(1 for d in out if d["cls_int"] == 0), 1)


# =============================================================================
#  §8  MOVEMENT COMPENSATION TESTS
# =============================================================================

@unittest.skipUnless(cv2 is not None and MovementEstimator is not None, "opencv/runtime deps missing")
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

    def _frame(self, shift_x: int = 0):
        rng = np.random.default_rng(42)
        base = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(600):
            x = int(rng.integers(5, 315))
            y = int(rng.integers(5, 315))
            cv2.circle(base, (x, y), 1, (255, 255, 255), -1)
        m = np.float32([[1, 0, shift_x], [0, 1, 0]])
        return cv2.warpAffine(base, m, (320, 320))

    def _vehicle(self, x1, y1, x2, y2):
        return [{"cls": "0", "top_left_x": x1, "top_left_y": y1,
                 "bottom_right_x": x2, "bottom_right_y": y2, "landing_status": "-1"}]

    def test_stationary_vehicle_with_camera_pan_marked_static(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()
        est.annotate(self._vehicle(100, 100, 140, 140), frame=self._frame(0))
        out = est.annotate(self._vehicle(112, 100, 152, 140), frame=self._frame(12))
        self.assertEqual(out[0]["motion_status"], "0")

    def test_actual_motion_preserved_with_compensation(self):
        Settings.MOTION_COMP_ENABLED = True
        est = MovementEstimator()
        est.annotate(self._vehicle(100, 100, 140, 140), frame=self._frame(0))
        out = est.annotate(self._vehicle(124, 100, 164, 140), frame=self._frame(12))
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


# =============================================================================
#  §9  FRAME DEDUP TESTS
# =============================================================================

@unittest.skipUnless(NetworkManager is not None and FrameFetchStatus is not None, "network deps missing")
class TestFrameDedup(unittest.TestCase):
    def setUp(self):
        self._orig = {"MAX_RETRIES": Settings.MAX_RETRIES, "SEEN_FRAME_LRU_SIZE": Settings.SEEN_FRAME_LRU_SIZE}
        Settings.MAX_RETRIES = 1
        Settings.SEEN_FRAME_LRU_SIZE = 2

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    def test_duplicate_frame_id_is_marked(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        r1, r2 = Mock(status_code=200), Mock(status_code=200)
        r1.json.return_value = {"id": "frame-1", "image_url": "/a.jpg"}
        r2.json.return_value = {"id": "frame-1", "image_url": "/a.jpg"}
        mgr.session.get = Mock(side_effect=[r1, r2])
        first = mgr.get_frame()
        second = mgr.get_frame()
        self.assertEqual(first.status, FrameFetchStatus.OK)
        self.assertFalse(first.is_duplicate)
        self.assertTrue(second.is_duplicate)

    def test_seen_frame_lru_evicts_oldest(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        self.assertFalse(mgr._mark_seen_frame("A"))
        self.assertFalse(mgr._mark_seen_frame("B"))
        self.assertFalse(mgr._mark_seen_frame("C"))
        self.assertFalse(mgr._mark_seen_frame("A"))


# =============================================================================
#  §10  IDEMPOTENCY SUBMIT TESTS
# =============================================================================

@unittest.skipUnless(NetworkManager is not None and SendResultStatus is not None, "network deps missing")
class TestIdempotencySubmit(unittest.TestCase):
    def setUp(self):
        self._orig = {"MAX_RETRIES": Settings.MAX_RETRIES, "IDEMPOTENCY_KEY_PREFIX": Settings.IDEMPOTENCY_KEY_PREFIX}
        Settings.MAX_RETRIES = 1
        Settings.IDEMPOTENCY_KEY_PREFIX = "aia"

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    def test_idempotency_header_is_sent(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.post = Mock(return_value=Mock(status_code=200))
        ok = mgr.send_result(
            frame_id="frame-7", detected_objects=[],
            detected_translation={"translation_x": 0, "translation_y": 0, "translation_z": 0},
            frame_data={"id": "frame-7", "url": "/f/7"}, frame_shape=None,
        )
        self.assertEqual(ok, SendResultStatus.ACKED)
        self.assertEqual(mgr.session.post.call_args.kwargs["headers"]["Idempotency-Key"], "aia:frame-7")

    def test_second_submit_same_frame_is_blocked(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.post = Mock(return_value=Mock(status_code=200))
        kw = dict(
            frame_id="frame-9", detected_objects=[],
            detected_translation={"translation_x": 0, "translation_y": 0, "translation_z": 0},
            frame_data={"id": "frame-9", "url": "/f/9"}, frame_shape=None,
        )
        first = mgr.send_result(**kw)
        second = mgr.send_result(**kw)
        self.assertEqual(first, SendResultStatus.ACKED)
        self.assertEqual(second, SendResultStatus.ACKED)
        self.assertEqual(mgr.session.post.call_count, 1)


# =============================================================================
#  §11  NETWORK TIMEOUT TESTS
# =============================================================================

@unittest.skipUnless(requests is not None and NetworkManager is not None, "network deps missing")
class TestNetworkTimeouts(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MAX_RETRIES": Settings.MAX_RETRIES, "REQUEST_TIMEOUT": Settings.REQUEST_TIMEOUT,
            "REQUEST_CONNECT_TIMEOUT_SEC": Settings.REQUEST_CONNECT_TIMEOUT_SEC,
            "REQUEST_READ_TIMEOUT_SEC_FRAME_META": Settings.REQUEST_READ_TIMEOUT_SEC_FRAME_META,
            "REQUEST_READ_TIMEOUT_SEC_IMAGE": Settings.REQUEST_READ_TIMEOUT_SEC_IMAGE,
            "REQUEST_READ_TIMEOUT_SEC_SUBMIT": Settings.REQUEST_READ_TIMEOUT_SEC_SUBMIT,
            "BACKOFF_BASE_SEC": Settings.BACKOFF_BASE_SEC,
            "BACKOFF_MAX_SEC": Settings.BACKOFF_MAX_SEC,
            "BACKOFF_JITTER_RATIO": Settings.BACKOFF_JITTER_RATIO,
        }
        Settings.MAX_RETRIES = 1
        Settings.REQUEST_TIMEOUT = 5
        Settings.REQUEST_CONNECT_TIMEOUT_SEC = 1.5
        Settings.REQUEST_READ_TIMEOUT_SEC_FRAME_META = 2.5
        Settings.REQUEST_READ_TIMEOUT_SEC_IMAGE = 4.0
        Settings.REQUEST_READ_TIMEOUT_SEC_SUBMIT = 3.5
        Settings.BACKOFF_BASE_SEC = 0.4
        Settings.BACKOFF_MAX_SEC = 5.0
        Settings.BACKOFF_JITTER_RATIO = 0.25

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    def test_timeout_tuple_is_used_per_endpoint(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        frame_resp, image_resp, submit_resp = Mock(status_code=204), Mock(status_code=500), Mock(status_code=200)
        get_calls = []

        def fake_get(url, **kwargs):
            get_calls.append((url, kwargs))
            return frame_resp if url.endswith(Settings.ENDPOINT_NEXT_FRAME) else image_resp

        mgr.session.get = fake_get
        mgr.session.post = Mock(return_value=submit_resp)
        mgr.get_frame()
        mgr.download_image({"frame_url": "/frame.jpg"})
        mgr.send_result(
            frame_id="f1", detected_objects=[],
            detected_translation={"translation_x": 0, "translation_y": 0, "translation_z": 0},
            frame_data={"id": "f1", "url": "/frame/1"}, frame_shape=None,
        )
        self.assertEqual(get_calls[0][1]["timeout"], (Settings.REQUEST_CONNECT_TIMEOUT_SEC, Settings.REQUEST_READ_TIMEOUT_SEC_FRAME_META))
        self.assertEqual(get_calls[1][1]["timeout"], (Settings.REQUEST_CONNECT_TIMEOUT_SEC, Settings.REQUEST_READ_TIMEOUT_SEC_IMAGE))
        self.assertEqual(mgr.session.post.call_args.kwargs["timeout"], (Settings.REQUEST_CONNECT_TIMEOUT_SEC, Settings.REQUEST_READ_TIMEOUT_SEC_SUBMIT))

    def test_backoff_delay_stays_in_expected_bounds(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        vals = [mgr._compute_backoff_delay(4) for _ in range(200)]
        assert all(2.4 <= v <= 4.0 for v in vals)
        large = [mgr._compute_backoff_delay(12) for _ in range(200)]
        assert all(3.75 <= v <= 5.0 for v in large)

    def test_timeout_counters_only_increment_on_timeout(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.get = Mock(side_effect=requests.Timeout("x"))
        result = mgr.get_frame()
        self.assertEqual(result.status.value, "transient_error")
        counts = mgr.consume_timeout_counters()
        self.assertEqual(counts["fetch"], 1)
        self.assertEqual(counts["image"], 0)


# =============================================================================
#  §12  NETWORK PAYLOAD GUARD TESTS
# =============================================================================

class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


@unittest.skipUnless(NetworkManager is not None, "network deps missing")
class TestNetworkPayloadGuard(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "RESULT_MAX_OBJECTS": Settings.RESULT_MAX_OBJECTS,
            "RESULT_CLASS_QUOTA": dict(Settings.RESULT_CLASS_QUOTA),
            "MAX_RETRIES": Settings.MAX_RETRIES,
        }
        Settings.RESULT_MAX_OBJECTS = 100
        Settings.RESULT_CLASS_QUOTA = {"0": 40, "1": 40, "2": 10, "3": 10}
        Settings.MAX_RETRIES = 3
        self.net = NetworkManager(base_url="http://localhost", simulation_mode=False)
        self.net._sleep_with_backoff = lambda attempt: None

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    @staticmethod
    def _obj(cls, conf, x, y):
        return {
            "cls": cls, "landing_status": "-1", "motion_status": "0",
            "top_left_x": x, "top_left_y": y,
            "bottom_right_x": x + 10, "bottom_right_y": y + 10, "_confidence": conf,
        }

    def test_limit_and_class_quota_are_enforced(self):
        objs = []
        for i in range(60):
            objs.append(self._obj("0", 0.99 - i * 0.001, i, i))
            objs.append(self._obj("1", 0.98 - i * 0.001, i, i + 1))
        for i in range(20):
            objs.append(self._obj("2", 0.97 - i * 0.001, i, i + 2))
            objs.append(self._obj("3", 0.96 - i * 0.001, i, i + 3))
        capped, stats = self.net._apply_object_caps(objs, frame_id="f-1")
        self.assertEqual(len(capped), 100)
        cc = {"0": 0, "1": 0, "2": 0, "3": 0}
        for d in capped:
            cc[d["cls"]] += 1
        self.assertLessEqual(cc["0"], 40)
        self.assertLessEqual(cc["1"], 40)
        self.assertGreater(stats["dropped_total"], 0)

    def test_capping_is_deterministic(self):
        src = [self._obj("0", 0.90, 5, 5), self._obj("0", 0.80, 6, 6),
               self._obj("1", 0.95, 4, 8), self._obj("2", 0.70, 1, 2), self._obj("3", 0.60, 3, 1)]
        a, _ = self.net._apply_object_caps(src + src, frame_id="f-2")
        b, _ = self.net._apply_object_caps(list(reversed(copy.deepcopy(src + src))), frame_id="f-2")
        self.assertEqual(a, b)

    def test_preflight_invalid_payload_forces_fallback(self):
        payload, rej, clip = self.net._preflight_validate_and_normalize_payload(
            payload={"id": 1, "user": "u"}, frame_shape=None, frame_id="f-3",
        )
        self.assertTrue(rej)
        self.assertFalse(clip)

    def test_4xx_then_fallback_200_returns_fallback_acked(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(400), _Response(200)])
        status = self.net.send_result(
            frame_id="f-4", detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
            frame_data={"id": "f-4", "user": "u", "url": "frame-url"}, frame_shape=(1080, 1920, 3),
        )
        self.assertEqual(status, SendResultStatus.FALLBACK_ACKED)

    def test_4xx_then_fallback_4xx_returns_permanent_rejected(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(422), _Response(400)])
        status = self.net.send_result(
            frame_id="f-5", detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
            frame_data={"id": "f-5", "user": "u", "url": "frame-url"}, frame_shape=(1080, 1920, 3),
        )
        self.assertEqual(status, SendResultStatus.PERMANENT_REJECTED)

    def test_5xx_retries_exhausted_returns_retryable_failure(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(500)] * 3)
        status = self.net.send_result(
            frame_id="f-6", detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
            frame_data={"id": "f-6", "user": "u", "url": "frame-url"}, frame_shape=(1080, 1920, 3),
        )
        self.assertEqual(status, SendResultStatus.RETRYABLE_FAILURE)


# =============================================================================
#  §13  COMPETITION LOOP HARDENING TESTS
# =============================================================================

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

    def consume_payload_guard_counters(self):
        return {"preflight_reject": 0, "payload_clipped": 0}

    def download_image(self, frame_data):
        _FakeNetwork.download_calls += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def send_result(self, frame_id, detected_objects, detected_translation,
                    frame_data=None, frame_shape=None, degrade=False,
                    detected_undefined_objects=None):
        _FakeNetwork.send_calls += 1
        return SendResultStatus.ACKED


@unittest.skipUnless(
    np is not None and FrameFetchResult is not None
    and FrameFetchStatus is not None and SendResultStatus is not None
    and main_module is not None,
    "runtime deps missing",
)
class TestCompetitionLoopHardening(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "DEBUG": Settings.DEBUG, "MAX_FRAMES": Settings.MAX_FRAMES,
            "LOOP_DELAY": Settings.LOOP_DELAY, "FPS_REPORT_INTERVAL": Settings.FPS_REPORT_INTERVAL,
            "DEGRADE_FETCH_ONLY_ENABLED": Settings.DEGRADE_FETCH_ONLY_ENABLED,
            "CB_SESSION_MAX_TRANSIENT_SEC": Settings.CB_SESSION_MAX_TRANSIENT_SEC,
            "CB_TRANSIENT_WINDOW_SEC": Settings.CB_TRANSIENT_WINDOW_SEC,
            "CB_TRANSIENT_MAX_EVENTS": Settings.CB_TRANSIENT_MAX_EVENTS,
            "CB_OPEN_COOLDOWN_SEC": Settings.CB_OPEN_COOLDOWN_SEC,
            "CB_MAX_OPEN_CYCLES": Settings.CB_MAX_OPEN_CYCLES,
        }
        Settings.DEBUG = False
        Settings.MAX_FRAMES = 50
        Settings.LOOP_DELAY = 0.0
        Settings.FPS_REPORT_INTERVAL = 99999
        Settings.DEGRADE_FETCH_ONLY_ENABLED = False
        # Resilience settings: yeterince büyük değerler ayarla
        Settings.CB_SESSION_MAX_TRANSIENT_SEC = 300.0
        Settings.CB_TRANSIENT_WINDOW_SEC = 60.0
        Settings.CB_TRANSIENT_MAX_EVENTS = 100
        Settings.CB_OPEN_COOLDOWN_SEC = 0.01
        Settings.CB_MAX_OPEN_CYCLES = 100
        _FakeNetwork.reset()
        self.summary_calls = []

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(Settings, k, v)

    def test_duplicate_frame_dropped_before_processing(self):
        _FakeNetwork.frame_results = [
            FrameFetchResult(status=FrameFetchStatus.OK, frame_data={"frame_id": "f1", "frame_url": "/f1.jpg", "gps_health": 1}, is_duplicate=False),
            FrameFetchResult(status=FrameFetchStatus.OK, frame_data={"frame_id": "f1", "frame_url": "/f1.jpg", "gps_health": 1}, is_duplicate=True),
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
        self.assertEqual(self.summary_calls[-1]["kpi_counters"]["frame_duplicate_drop"], 1)

    def test_transient_fetch_timeout_recovers(self):
        _FakeNetwork.frame_results = [
            FrameFetchResult(status=FrameFetchStatus.TRANSIENT_ERROR, error_type="retries_exhausted"),
            FrameFetchResult(status=FrameFetchStatus.OK, frame_data={"frame_id": "f2", "frame_url": "/f2.jpg", "gps_health": 1}, is_duplicate=False),
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
        self.assertEqual(self.summary_calls[-1]["kpi_counters"]["timeout_fetch"], 1)
