import cv2
import numpy as np
from typing import Optional
from config.settings import Settings

class FrameContext:
    """Frame için ortak hesaplamalar (gray conversion). Detection, movement, localization tekrar hesaplamasın."""
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self._gray: Optional[np.ndarray] = None

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return self._gray
