import unittest
from src.send_state import apply_send_result_status

class MockStatus:
    def __init__(self, value):
        self.value = value

class TestSendState(unittest.TestCase):
    def setUp(self):
        self.pending = {"frame_id": "test-frame"}
        self.counters = {
            "send_ok": 0,
            "send_fail": 0,
            "send_fallback_ok": 0,
            "send_permanent_reject": 0
        }

    def test_apply_status_acked_string(self):
        next_pending, should_abort, success = apply_send_result_status(
            "acked", self.pending, self.counters
        )
        self.assertIsNone(next_pending)
        self.assertFalse(should_abort)
        self.assertTrue(success)
        self.assertEqual(self.counters["send_ok"], 1)
        self.assertEqual(self.counters["send_fail"], 0)

    def test_apply_status_acked_enum(self):
        next_pending, should_abort, success = apply_send_result_status(
            MockStatus("acked"), self.pending, self.counters
        )
        self.assertIsNone(next_pending)
        self.assertFalse(should_abort)
        self.assertTrue(success)
        self.assertEqual(self.counters["send_ok"], 1)

    def test_apply_status_fallback_acked(self):
        next_pending, should_abort, success = apply_send_result_status(
            "fallback_acked", self.pending, self.counters
        )
        self.assertIsNone(next_pending)
        self.assertFalse(should_abort)
        self.assertTrue(success)
        self.assertEqual(self.counters["send_ok"], 1)
        self.assertEqual(self.counters["send_fallback_ok"], 1)

    def test_apply_status_permanent_rejected(self):
        next_pending, should_abort, success = apply_send_result_status(
            "permanent_rejected", self.pending, self.counters
        )
        self.assertIs(next_pending, self.pending)
        self.assertTrue(should_abort)
        self.assertFalse(success)
        self.assertEqual(self.counters["send_fail"], 1)
        self.assertEqual(self.counters["send_permanent_reject"], 1)

    def test_apply_status_retryable_failure(self):
        next_pending, should_abort, success = apply_send_result_status(
            "retryable_failure", self.pending, self.counters
        )
        self.assertIs(next_pending, self.pending)
        self.assertFalse(should_abort)
        self.assertFalse(success)
        self.assertEqual(self.counters["send_fail"], 1)
        self.assertEqual(self.counters["send_ok"], 0)

    def test_apply_status_unknown(self):
        next_pending, should_abort, success = apply_send_result_status(
            "something_else", self.pending, self.counters
        )
        self.assertIs(next_pending, self.pending)
        self.assertFalse(should_abort)
        self.assertFalse(success)
        self.assertEqual(self.counters["send_fail"], 1)

if __name__ == "__main__":
    unittest.main()
