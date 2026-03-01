import cv2
import numpy as np
from typing import Optional
from config.settings import Settings

class FrameContext:
    """
    Centralized object to compute common frame transformations (like grayscale)
    and shared features (like optical flow) to eliminate redundant work across modules.
    """
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self._gray: Optional[np.ndarray] = None

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return self._gray
