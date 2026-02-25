import unittest

from config.settings import Settings
from src.resilience import ResilienceState, SessionResilienceController


class _StubLog:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warn(self, msg):
        self.lines.append(("warn", msg))

    def error(self, msg):
        self.lines.append(("error", msg))


class TestSessionResilienceController(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "CB_TRANSIENT_WINDOW_SEC": Settings.CB_TRANSIENT_WINDOW_SEC,
            "CB_TRANSIENT_MAX_EVENTS": Settings.CB_TRANSIENT_MAX_EVENTS,
            "CB_OPEN_COOLDOWN_SEC": Settings.CB_OPEN_COOLDOWN_SEC,
            "CB_MAX_OPEN_CYCLES": Settings.CB_MAX_OPEN_CYCLES,
            "CB_SESSION_MAX_TRANSIENT_SEC": Settings.CB_SESSION_MAX_TRANSIENT_SEC,
        }
        Settings.CB_TRANSIENT_WINDOW_SEC = 2.0
        Settings.CB_TRANSIENT_MAX_EVENTS = 3
        Settings.CB_OPEN_COOLDOWN_SEC = 0.2
        Settings.CB_MAX_OPEN_CYCLES = 2
        Settings.CB_SESSION_MAX_TRANSIENT_SEC = 0.4
        self.log = _StubLog()

    def tearDown(self):
        for key, value in self._orig.items():
            setattr(Settings, key, value)

    def test_transient_window_opens_breaker(self):
        c = SessionResilienceController(self.log)

        c.on_fetch_transient()
        c.on_fetch_transient()
        self.assertEqual(c.state, ResilienceState.DEGRADED)

        c.on_fetch_transient()
        self.assertEqual(c.state, ResilienceState.OPEN)
        self.assertEqual(c.stats.breaker_open_count, 1)

    def test_open_to_half_open_to_normal_recovery(self):
        c = SessionResilienceController(self.log)
        c.on_fetch_transient()
        c.on_fetch_transient()
        c.on_fetch_transient()
        self.assertEqual(c.state, ResilienceState.OPEN)

        self.assertFalse(c.before_fetch())
        # Cooldown dolduktan sonra half-open benzeri DEGRADED'a geçer.
        import time
        time.sleep(0.25)
        self.assertTrue(c.before_fetch())
        self.assertEqual(c.state, ResilienceState.DEGRADED)

        c.on_success_cycle()
        self.assertEqual(c.state, ResilienceState.NORMAL)
        self.assertGreaterEqual(c.stats.recovered_count, 1)

    def test_session_wall_clock_abort(self):
        c = SessionResilienceController(self.log)
        c.on_ack_failure()
        self.assertEqual(c.state, ResilienceState.DEGRADED)

        import time
        time.sleep(0.45)
        reason = c.should_abort()
        self.assertIsNotNone(reason)
        self.assertIn("Transient wall time exceeded", reason)

    def test_breaker_open_cycles_abort(self):
        c = SessionResilienceController(self.log)
        c.stats.breaker_open_count = 3
        reason = c.should_abort()
        self.assertIsNotNone(reason)
        self.assertIn("Breaker open cycles exceeded", reason)


if __name__ == "__main__":
    unittest.main()
