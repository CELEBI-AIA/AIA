import unittest

from src.send_state import apply_send_result_status


class TestMainAckStateMachine(unittest.TestCase):
    def test_retryable_failure_keeps_pending_result(self):
        pending = {"frame_id": "f-1"}
        counters = {"send_ok": 0, "send_fail": 0, "send_fallback_ok": 0, "send_permanent_reject": 0}

        next_pending, should_abort, success_cycle = apply_send_result_status(
            send_status="retryable_failure",
            pending_result=pending,
            kpi_counters=counters,
        )

        self.assertIs(next_pending, pending)
        self.assertFalse(should_abort)
        self.assertFalse(success_cycle)
        self.assertEqual(counters["send_fail"], 1)

    def test_fallback_acked_clears_pending_and_counts(self):
        pending = {"frame_id": "f-2"}
        counters = {"send_ok": 0, "send_fail": 0, "send_fallback_ok": 0, "send_permanent_reject": 0}

        next_pending, should_abort, success_cycle = apply_send_result_status(
            send_status="fallback_acked",
            pending_result=pending,
            kpi_counters=counters,
        )

        self.assertIsNone(next_pending)
        self.assertFalse(should_abort)
        self.assertTrue(success_cycle)
        self.assertEqual(counters["send_ok"], 1)
        self.assertEqual(counters["send_fallback_ok"], 1)

    def test_permanent_rejected_aborts_session(self):
        pending = {"frame_id": "f-3"}
        counters = {"send_ok": 0, "send_fail": 0, "send_fallback_ok": 0, "send_permanent_reject": 0}

        next_pending, should_abort, success_cycle = apply_send_result_status(
            send_status="permanent_rejected",
            pending_result=pending,
            kpi_counters=counters,
        )

        self.assertIs(next_pending, pending)
        self.assertTrue(should_abort)
        self.assertFalse(success_cycle)
        self.assertEqual(counters["send_fail"], 1)
        self.assertEqual(counters["send_permanent_reject"], 1)


if __name__ == "__main__":
    unittest.main()
