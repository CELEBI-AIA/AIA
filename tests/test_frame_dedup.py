import unittest
from unittest.mock import Mock

from config.settings import Settings
from src.network import FrameFetchStatus, NetworkManager


class TestFrameDedup(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "MAX_RETRIES": Settings.MAX_RETRIES,
            "SEEN_FRAME_LRU_SIZE": Settings.SEEN_FRAME_LRU_SIZE,
        }
        Settings.MAX_RETRIES = 1
        Settings.SEEN_FRAME_LRU_SIZE = 2

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    def test_duplicate_frame_id_is_marked(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)
        resp1 = Mock(status_code=200)
        resp2 = Mock(status_code=200)
        resp1.json.return_value = {"id": "frame-1", "image_url": "/a.jpg"}
        resp2.json.return_value = {"id": "frame-1", "image_url": "/a.jpg"}
        mgr.session.get = Mock(side_effect=[resp1, resp2])

        first = mgr.get_frame()
        second = mgr.get_frame()

        self.assertEqual(first.status, FrameFetchStatus.OK)
        self.assertFalse(first.is_duplicate)
        self.assertEqual(second.status, FrameFetchStatus.OK)
        self.assertTrue(second.is_duplicate)

    def test_seen_frame_lru_evicts_oldest(self):
        mgr = NetworkManager(base_url="http://test", simulation_mode=False)

        self.assertFalse(mgr._mark_seen_frame("A"))
        self.assertFalse(mgr._mark_seen_frame("B"))
        self.assertFalse(mgr._mark_seen_frame("C"))  # A evicted (size=2)
        self.assertFalse(mgr._mark_seen_frame("A"))  # Not duplicate anymore


if __name__ == "__main__":
    unittest.main()
