import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest  # noqa: F401 - pytest framework

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
mock_torchvision = MagicMock()
mock_ultralytics = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torchvision'] = mock_torchvision
sys.modules['ultralytics'] = mock_ultralytics
