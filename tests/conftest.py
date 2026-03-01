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


# pytest-timeout: pytest.ini üzerinden timeout=10
