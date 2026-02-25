import unittest
from unittest.mock import Mock

from config.settings import Settings
from src.network import NetworkManager


class TestIdempotencySubmit(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MAX_RETRIES": Settings.MAX_RETRIES,
            "IDEMPOTENCY_KEY_PREFIX": Settings.IDEMPOTENCY_KEY_PREFIX,
        }
        Settings.MAX_RETRIES = 1
        Settings.IDEMPOTENCY_KEY_PREFIX = "aia"

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    def test_idempotency_header_is_sent(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.post = Mock(return_value=Mock(status_code=200))

        ok = mgr.send_result(
            frame_id="frame-7",
            detected_objects=[],
            detected_translation={"translation_x": 0.0, "translation_y": 0.0, "translation_z": 0.0},
            frame_data={"id": "frame-7", "url": "/f/7"},
            frame_shape=None,
        )

        self.assertTrue(ok)
        headers = mgr.session.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Idempotency-Key"], "aia:frame-7")

    def test_second_submit_same_frame_is_blocked(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.post = Mock(return_value=Mock(status_code=200))

        first = mgr.send_result(
            frame_id="frame-9",
            detected_objects=[],
            detected_translation={"translation_x": 0.0, "translation_y": 0.0, "translation_z": 0.0},
            frame_data={"id": "frame-9", "url": "/f/9"},
            frame_shape=None,
        )
        second = mgr.send_result(
            frame_id="frame-9",
            detected_objects=[],
            detected_translation={"translation_x": 0.0, "translation_y": 0.0, "translation_z": 0.0},
            frame_data={"id": "frame-9", "url": "/f/9"},
            frame_shape=None,
        )

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(mgr.session.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
