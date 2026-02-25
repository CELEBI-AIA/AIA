import unittest
from unittest.mock import Mock

from config.settings import Settings
try:
    import requests
except ImportError:  # pragma: no cover - environment-dependent
    requests = None
try:
    from src.network import NetworkManager
except Exception:  # pragma: no cover - environment-dependent
    NetworkManager = None


@unittest.skipUnless(requests is not None and NetworkManager is not None, "network deps are missing")
class TestNetworkTimeouts(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MAX_RETRIES": Settings.MAX_RETRIES,
            "REQUEST_TIMEOUT": Settings.REQUEST_TIMEOUT,
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
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    def test_timeout_tuple_is_used_per_endpoint(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)

        frame_resp = Mock(status_code=204)
        image_resp = Mock(status_code=500)
        submit_resp = Mock(status_code=200)

        get_calls = []

        def fake_get(url, **kwargs):
            get_calls.append((url, kwargs))
            if url.endswith(Settings.ENDPOINT_NEXT_FRAME):
                return frame_resp
            return image_resp

        mgr.session.get = fake_get
        mgr.session.post = Mock(return_value=submit_resp)

        mgr.get_frame()
        mgr.download_image({"frame_url": "/frame.jpg"})
        mgr.send_result(
            frame_id="f1",
            detected_objects=[],
            detected_translation={"translation_x": 0.0, "translation_y": 0.0, "translation_z": 0.0},
            frame_data={"id": "f1", "url": "/frame/1"},
            frame_shape=None,
        )

        self.assertEqual(
            get_calls[0][1]["timeout"],
            (Settings.REQUEST_CONNECT_TIMEOUT_SEC, Settings.REQUEST_READ_TIMEOUT_SEC_FRAME_META),
        )
        self.assertEqual(
            get_calls[1][1]["timeout"],
            (Settings.REQUEST_CONNECT_TIMEOUT_SEC, Settings.REQUEST_READ_TIMEOUT_SEC_IMAGE),
        )
        self.assertEqual(
            mgr.session.post.call_args.kwargs["timeout"],
            (Settings.REQUEST_CONNECT_TIMEOUT_SEC, Settings.REQUEST_READ_TIMEOUT_SEC_SUBMIT),
        )

    def test_backoff_delay_stays_in_expected_bounds(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)

        # attempt=4 -> base*2^(n-1)=3.2s, jitter %25 => [2.4, 4.0]
        values = [mgr._compute_backoff_delay(4) for _ in range(200)]
        self.assertTrue(all(2.4 <= val <= 4.0 for val in values))

        # Large attempts should stay bounded by max delay.
        large_values = [mgr._compute_backoff_delay(12) for _ in range(200)]
        self.assertTrue(all(3.75 <= val <= 5.0 for val in large_values))

    def test_timeout_counters_only_increment_on_timeout(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        mgr.session.get = Mock(side_effect=requests.Timeout("x"))

        result = mgr.get_frame()
        self.assertEqual(result.status.value, "transient_error")

        counts = mgr.consume_timeout_counters()
        self.assertEqual(counts["fetch"], 1)
        self.assertEqual(counts["image"], 0)
        self.assertEqual(counts["submit"], 0)


if __name__ == "__main__":
    unittest.main()
