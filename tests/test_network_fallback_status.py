"""Network fallback flow regression tests."""

import sys
from unittest.mock import MagicMock, Mock

sys.modules.setdefault("cv2", MagicMock())

from config.settings import Settings
from src.network import NetworkManager, SendResultStatus


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class TestNetworkFallbackStatusCodes:
    def setup_method(self):
        self.orig_max_retries = Settings.MAX_RETRIES
        self.orig_payload_adapter_version = Settings.PAYLOAD_ADAPTER_VERSION
        Settings.MAX_RETRIES = 3
        Settings.PAYLOAD_ADAPTER_VERSION = "v1"
        self.net = NetworkManager(base_url="http://localhost", simulation_mode=False)
        self.net._sleep_with_backoff = lambda attempt: None

    def teardown_method(self):
        Settings.MAX_RETRIES = self.orig_max_retries
        Settings.PAYLOAD_ADAPTER_VERSION = self.orig_payload_adapter_version

    def test_preflight_reject_path_still_returns_fallback_acked(self):
        self.net.session = Mock()
        self.net.session.post = Mock(return_value=_Response(200))

        status = self.net.send_result(
            frame_id="frame-preflight",
            detected_objects=[{
                "cls": "2",
                "motion_status": "0",
                "top_left_x": 1,
                "top_left_y": 1,
                "bottom_right_x": 10,
                "bottom_right_y": 10,
            }],
            detected_translation={
                "translation_x": "unknown",
                "translation_y": None,
                "translation_z": "NaN",
            },
            frame_data={"id": "frame-preflight", "user": "u", "url": "frame-url"},
            frame_shape=(1080, 1920, 3),
            degrade=True,
        )

        assert status == SendResultStatus.FALLBACK_ACKED
        sent_payload = self.net.session.post.call_args.kwargs["json"]
        assert sent_payload["detected_translations"][0] == {
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0,
        }

    def test_4xx_then_fallback_still_returns_fallback_acked(self):
        self.net.session = Mock()
        self.net.session.post = Mock(side_effect=[_Response(400), _Response(200)])

        status = self.net.send_result(
            frame_id="frame-4xx",
            detected_objects=[{"cls": "0", "confidence": 0.9}],
            detected_translation={
                "translation_x": 1,
                "translation_y": 2,
                "translation_z": 3,
            },
            frame_data={"id": "frame-4xx", "user": "u", "url": "frame-url"},
            frame_shape=(1080, 1920, 3),
        )

        assert status == SendResultStatus.FALLBACK_ACKED
        assert self.net.session.post.call_count == 2

    def test_safe_fallback_payload_handles_invalid_translation_container(self):
        payload = {
            "id": "frame-invalid-trans",
            "user": "u",
            "frame": "frame-url",
            "detected_objects": [],
            "detected_translations": ["not-a-dict"],
        }

        out = self.net._build_safe_fallback_payload(payload)

        assert out["detected_translations"][0] == {
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0,
        }
