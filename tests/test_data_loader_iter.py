"""DatasetLoader video iteration behavior tests."""

import sys
import types
from unittest.mock import Mock

import numpy as np

# Inject a lightweight cv2 stub before importing project modules.
cv2_stub = types.SimpleNamespace(
    CAP_PROP_POS_FRAMES=1,
    CAP_PROP_FRAME_COUNT=2,
)
sys.modules.setdefault("cv2", cv2_stub)

from src.data_loader import DatasetLoader  # noqa: E402


class FakeVideoCapture:
    def __init__(self, frame_count, reset_success=True):
        self.frame_count = frame_count
        self.reset_success = reset_success
        self.pos = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop != cv2_stub.CAP_PROP_POS_FRAMES:
            return False
        if value == 0 and self.reset_success:
            self.pos = 0
            return True
        return False

    def get(self, prop):
        if prop == cv2_stub.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        if prop == cv2_stub.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        return 0.0

    def release(self):
        return None

    def read(self):
        if self.pos >= self.frame_count:
            return False, None
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.pos += 1
        return True, frame


def build_loader(capture):
    loader = DatasetLoader.__new__(DatasetLoader)
    loader.log = Mock()
    loader._frames = []
    loader._video_capture = capture
    loader._video_total_frames = capture.frame_count
    loader._index = 0
    loader._mode = "vid"
    loader._sequence_name = "fake"
    return loader


def test_video_loader_can_be_reiterated_from_start():
    capture = FakeVideoCapture(frame_count=3, reset_success=True)
    loader = build_loader(capture)

    first_pass = [item["frame_idx"] for item in loader]
    second_pass = [item["frame_idx"] for item in loader]

    assert first_pass == [0, 1, 2]
    assert second_pass == [0, 1, 2]


def test_iter_warns_when_video_reset_fails():
    capture = FakeVideoCapture(frame_count=2, reset_success=False)
    loader = build_loader(capture)
    capture.pos = 1

    iter(loader)

    loader.log.warn.assert_called_once()


def test_len_fallback_when_video_metadata_missing():
    loader = DatasetLoader.__new__(DatasetLoader)
    loader.log = Mock()
    loader._frames = []
    loader._video_capture = FakeVideoCapture(frame_count=4, reset_success=True)
    loader._video_total_frames = 0
    loader._index = 0
    loader._mode = "vid"
    loader._sequence_name = "fake"

    count = loader._estimate_video_frame_count()
    loader._video_total_frames = count

    assert count == 4
    assert len(loader) == 4


def test_zero_frame_video_metadata_fallback_results_in_empty_iteration():
    loader = DatasetLoader.__new__(DatasetLoader)
    loader.log = Mock()
    loader._frames = []
    loader._video_capture = FakeVideoCapture(frame_count=0, reset_success=True)
    loader._video_total_frames = 0
    loader._index = 0
    loader._mode = "vid"
    loader._sequence_name = "fake"

    count = loader._estimate_video_frame_count()
    loader._video_total_frames = count

    assert count == 0
    assert loader.is_ready is False
    assert len(loader) == 0
    assert list(loader) == []
