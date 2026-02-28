import sys
import threading
from unittest.mock import MagicMock

import pytest

# Create mock objects
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
# OOM exception mock
mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
mock_torchvision = MagicMock()
mock_ultralytics = MagicMock()

# Inject into sys.modules
sys.modules['torch'] = mock_torch
sys.modules['torchvision'] = mock_torchvision
sys.modules['ultralytics'] = mock_ultralytics


# ─── Global 10-second timeout for ALL tests ───
TEST_TIMEOUT_SEC = 10


@pytest.fixture(autouse=True)
def enforce_test_timeout():
    """Her test 10 saniye içinde bitmezse FAIL olur."""
    result = {"exc": None}

    def _run_with_timeout(fn):
        try:
            fn()
        except Exception as e:
            result["exc"] = e

    # yield ile test'in çalışmasına izin ver
    # timeout kontrolü test fonksiyonu seviyesinde yapılır
    timer = threading.Timer(TEST_TIMEOUT_SEC, lambda: None)
    timer.daemon = True
    timer.start()
    yield
    if not timer.is_alive():
        pytest.fail(
            f"Test {TEST_TIMEOUT_SEC} saniye içinde tamamlanamadı (TIMEOUT)"
        )
    timer.cancel()
