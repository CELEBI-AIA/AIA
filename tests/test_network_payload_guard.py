import copy
import sys
import types
import unittest
from unittest.mock import Mock

try:  # pragma: no cover - environment dependent
    import cv2  # type: ignore # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    sys.modules["cv2"] = types.SimpleNamespace()
try:  # pragma: no cover - environment dependent
    import requests  # type: ignore # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    class _Timeout(Exception):
        pass

    class _ConnectionError(Exception):
        pass

    class _Session:
        def post(self, *args, **kwargs):
            raise NotImplementedError

    sys.modules["requests"] = types.SimpleNamespace(
        Session=_Session,
        Timeout=_Timeout,
        ConnectionError=_ConnectionError,
    )

from config.settings import Settings
from src.network import NetworkManager, SendResultStatus


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code


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
        Settings.RESULT_MAX_OBJECTS = self._orig["RESULT_MAX_OBJECTS"]
        Settings.RESULT_CLASS_QUOTA = self._orig["RESULT_CLASS_QUOTA"]
        Settings.MAX_RETRIES = self._orig["MAX_RETRIES"]

    @staticmethod
    def _obj(cls: str, conf: float, x: int, y: int):
        return {
            "cls": cls,
            "landing_status": "-1",
            "motion_status": "0",
            "top_left_x": x,
            "top_left_y": y,
            "bottom_right_x": x + 10,
            "bottom_right_y": y + 10,
            "_confidence": conf,
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
        cls_counts = {"0": 0, "1": 0, "2": 0, "3": 0}
        for det in capped:
            cls_counts[det["cls"]] += 1

        self.assertLessEqual(cls_counts["0"], 40)
        self.assertLessEqual(cls_counts["1"], 40)
        self.assertLessEqual(cls_counts["2"], 10)
        self.assertLessEqual(cls_counts["3"], 10)
        self.assertGreater(stats["dropped_total"], 0)

    def test_capping_is_deterministic(self):
        source = [
            self._obj("0", 0.90, 5, 5),
            self._obj("0", 0.80, 6, 6),
            self._obj("1", 0.95, 4, 8),
            self._obj("2", 0.70, 1, 2),
            self._obj("3", 0.60, 3, 1),
        ]
        objs_a = source + source
        objs_b = list(reversed(copy.deepcopy(objs_a)))

        out_a, _ = self.net._apply_object_caps(objs_a, frame_id="f-2")
        out_b, _ = self.net._apply_object_caps(objs_b, frame_id="f-2")

        self.assertEqual(out_a, out_b)

    def test_preflight_invalid_payload_forces_fallback(self):
        payload, preflight_rejected, clipped = self.net._preflight_validate_and_normalize_payload(
            payload={"id": 1, "user": "u"},
            frame_shape=None,
            frame_id="f-3",
        )

        self.assertTrue(preflight_rejected)
        self.assertFalse(clipped)
        self.assertEqual(payload["detected_objects"], [])
        self.assertEqual(payload["detected_translations"][0]["translation_x"], 0.0)

    def test_4xx_then_fallback_200_returns_fallback_acked(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(400), _Response(200)])

        status = self.net.send_result(
            frame_id="f-4",
            detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
            frame_data={"id": "f-4", "user": "u", "url": "frame-url"},
            frame_shape=(1080, 1920, 3),
        )

        self.assertEqual(status, SendResultStatus.FALLBACK_ACKED)
        self.assertEqual(self.net.session.post.call_count, 2)
        first_payload = self.net.session.post.call_args_list[0].kwargs["json"]
        second_payload = self.net.session.post.call_args_list[1].kwargs["json"]
        self.assertGreaterEqual(len(first_payload["detected_objects"]), 1)
        self.assertEqual(second_payload["detected_objects"], [])

    def test_4xx_then_fallback_4xx_returns_permanent_rejected(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(422), _Response(400)])

        status = self.net.send_result(
            frame_id="f-5",
            detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
            frame_data={"id": "f-5", "user": "u", "url": "frame-url"},
            frame_shape=(1080, 1920, 3),
        )

        self.assertEqual(status, SendResultStatus.PERMANENT_REJECTED)

    def test_5xx_retries_exhausted_returns_retryable_failure(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(500), _Response(500), _Response(500)])

        status = self.net.send_result(
            frame_id="f-6",
            detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
            frame_data={"id": "f-6", "user": "u", "url": "frame-url"},
            frame_shape=(1080, 1920, 3),
        )

        self.assertEqual(status, SendResultStatus.RETRYABLE_FAILURE)
        self.assertEqual(self.net.session.post.call_count, Settings.MAX_RETRIES)


if __name__ == "__main__":
    unittest.main()
