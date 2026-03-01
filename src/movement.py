"""Taşıt hareketlilik (motion_status): 1=hareketli, 0=sabit, -1=taşıt değil.
Merkez takibi + kamera kayması kompanzasyonu ile yer değiştirme hesaplanır."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.frame_context import FrameContext

import cv2
import numpy as np
from config.settings import Settings


@dataclass
class _Track:
    history: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)
    missed: int = 0


class MovementEstimator:
    """Merkez takibi ile taşıt hareket durumu (1=hareketli, 0=sabit, -1=taşıt değil)."""

    def __init__(self) -> None:
        self._tracks: Dict[int, _Track] = {}
        self._next_track_id: int = 1
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._cam_shift_hist: Deque[Tuple[float, float]] = deque(
            maxlen=Settings.MOVEMENT_WINDOW_FRAMES
        )
        self._cam_total_x: float = 0.0
        self._cam_total_y: float = 0.0
        self._frame_width: int = Settings.MOVEMENT_THRESHOLD_REF_WIDTH
        self._is_frozen_frame: bool = False
        self._frame_diff: float = float("inf")

    def annotate(self, detections: List[Dict], frame_ctx: Optional["FrameContext"] = None) -> List[Dict]:
        if frame_ctx is not None:
            if isinstance(frame_ctx, np.ndarray):
                self._frame_width = frame_ctx.shape[1]
            else:
                self._frame_width = frame_ctx.frame.shape[1]

        cam_dx = cam_dy = 0.0
        if Settings.MOTION_COMP_ENABLED and frame_ctx is not None:
            cam_dx, cam_dy = self._estimate_camera_shift(frame_ctx)
        self._cam_shift_hist.append((cam_dx, cam_dy))
        self._cam_total_x += cam_dx
        self._cam_total_y += cam_dy

        _CAM_TOTAL_CAP = 1e6
        self._cam_total_x = max(-_CAM_TOTAL_CAP, min(_CAM_TOTAL_CAP, self._cam_total_x))
        self._cam_total_y = max(-_CAM_TOTAL_CAP, min(_CAM_TOTAL_CAP, self._cam_total_y))

        self._is_frozen_frame = self._frame_diff < Settings.FROZEN_FRAME_DIFF_THRESHOLD

        vehicles: List[Tuple[int, Dict]] = []
        for idx, det in enumerate(detections):
            cls_val = det.get("cls_int", det.get("cls"))
            if cls_val == Settings.CLASS_TASIT or (isinstance(cls_val, str) and cls_val == "0"):
                vehicles.append((idx, det))
            else:
                det["motion_status"] = "-1"

        if not vehicles:
            self._age_tracks(set())
            return detections

        centers = {idx: self._center(det) for idx, det in vehicles}
        assignments = self._match(centers)
        matched_track_ids = set(assignments.values())
        self._age_tracks(matched_track_ids)

        for idx, det in vehicles:
            track_id = assignments.get(idx)
            if track_id is None:
                track_id = self._create_track(centers[idx])
            track = self._tracks[track_id]
            cx, cy = centers[idx]

            if not self._is_frozen_frame:
                track.history.append((cx, cy, self._cam_total_x, self._cam_total_y))
            track.missed = 0

            status = self._status(track.history)
            det["motion_status"] = status

        return detections

    def _status(self, history: Deque[Tuple[float, float, float, float]]) -> str:
        """Sliding window: kamera-kompanze edilmiş yer değiştirme > threshold → hareketli"""
        n = len(history)
        scale = self._frame_width / Settings.MOVEMENT_THRESHOLD_REF_WIDTH
        threshold = Settings.MOVEMENT_THRESHOLD_PX * scale

        if n < Settings.MOVEMENT_MIN_HISTORY:
            if n >= 2:
                x0, y0, cam0_x, cam0_y = history[0]
                x1, y1, cam1_x, cam1_y = history[-1]
                rel_dx = (x1 - x0) - (cam1_x - cam0_x)
                rel_dy = (y1 - y0) - (cam1_y - cam0_y)
                dist = (rel_dx * rel_dx + rel_dy * rel_dy) ** 0.5
                if dist >= threshold * 0.9:
                    return "1"
            return "0"

        step = max(1, Settings.MOVEMENT_MIN_HISTORY - 1)

        for i in range(n - step):
            j = i + step
            x0, y0, cam0_x, cam0_y = history[i]
            x1, y1, cam1_x, cam1_y = history[j]

            rel_dx = (x1 - x0) - (cam1_x - cam0_x)
            rel_dy = (y1 - y0) - (cam1_y - cam0_y)
            dist = (rel_dx * rel_dx + rel_dy * rel_dy) ** 0.5

            if dist >= threshold:
                return "1"

        return "0"

    def _match(self, centers: Dict[int, Tuple[float, float]]) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        if not self._tracks:
            return assignments

        candidates: List[Tuple[float, int, int]] = []
        for det_idx, (cx, cy) in centers.items():
            for track_id, track in self._tracks.items():
                if not track.history:
                    continue
                tx, ty = track.history[-1][:2]
                dx = cx - tx
                dy = cy - ty
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= Settings.MOVEMENT_MATCH_DISTANCE_PX:
                    candidates.append((dist, det_idx, track_id))

        used_dets = set()
        used_tracks = set()
        for _, det_idx, track_id in sorted(candidates, key=lambda item: item[0]):
            if det_idx in used_dets or track_id in used_tracks:
                continue
            assignments[det_idx] = track_id
            used_dets.add(det_idx)
            used_tracks.add(track_id)
        return assignments

    def _age_tracks(self, matched_track_ids: set) -> None:
        to_delete: List[int] = []
        for track_id in list(self._tracks.keys()):
            track = self._tracks[track_id]
            if track_id not in matched_track_ids:
                track.missed += 1
                if track.missed > Settings.MOVEMENT_MAX_MISSED_FRAMES:
                    to_delete.append(track_id)
        for track_id in to_delete:
            self._tracks[track_id].history.clear()
            del self._tracks[track_id]

    def _create_track(self, center: Tuple[float, float]) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        track = _Track(history=deque(maxlen=Settings.MOVEMENT_WINDOW_FRAMES))
        cx, cy = center
        track.history.append((cx, cy, self._cam_total_x, self._cam_total_y))
        self._tracks[track_id] = track
        return track_id

    @staticmethod
    def _center(det: Dict) -> Tuple[float, float]:
        return (
            (float(det.get("top_left_x", 0)) + float(det.get("bottom_right_x", 0))) / 2.0,
            (float(det.get("top_left_y", 0)) + float(det.get("bottom_right_y", 0))) / 2.0,
        )

    def _estimate_camera_shift(self, frame_ctx: "FrameContext") -> Tuple[float, float]:
        if isinstance(frame_ctx, np.ndarray):
            from src.frame_context import FrameContext
            frame_ctx = FrameContext(frame_ctx)
        gray = frame_ctx.gray
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=Settings.MOTION_COMP_MAX_CORNERS,
                qualityLevel=Settings.MOTION_COMP_QUALITY_LEVEL,
                minDistance=Settings.MOTION_COMP_MIN_DISTANCE,
            )
            self._frame_diff = float("inf")
            return 0.0, 0.0

        self._frame_diff = float(cv2.absdiff(self._prev_gray, gray).mean())

        if self._prev_points is None or len(self._prev_points) < Settings.MOTION_COMP_MIN_FEATURES:
            self._prev_points = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=Settings.MOTION_COMP_MAX_CORNERS,
                qualityLevel=Settings.MOTION_COMP_QUALITY_LEVEL,
                minDistance=Settings.MOTION_COMP_MIN_DISTANCE,
            )
            if self._prev_points is None or len(self._prev_points) < 5:
                self._prev_gray = gray
                self._prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=Settings.MOTION_COMP_MAX_CORNERS,
                    qualityLevel=Settings.MOTION_COMP_QUALITY_LEVEL,
                    minDistance=Settings.MOTION_COMP_MIN_DISTANCE,
                )
                return 0.0, 0.0

        if self._prev_points is None:
            return 0.0, 0.0

        win = Settings.MOTION_COMP_WIN_SIZE
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._prev_points,
            None,
            winSize=(win, win),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            self._prev_gray = gray
            self._prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=Settings.MOTION_COMP_MAX_CORNERS,
                qualityLevel=Settings.MOTION_COMP_QUALITY_LEVEL,
                minDistance=Settings.MOTION_COMP_MIN_DISTANCE,
            )
            return 0.0, 0.0

        valid = status.flatten() == 1
        old = self._prev_points[valid].reshape(-1, 2)
        new = next_pts[valid].reshape(-1, 2)
        if len(new) < 5:
            self._prev_gray = gray
            self._prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=Settings.MOTION_COMP_MAX_CORNERS,
                qualityLevel=Settings.MOTION_COMP_QUALITY_LEVEL,
                minDistance=Settings.MOTION_COMP_MIN_DISTANCE,
            )
            return 0.0, 0.0

        deltas = new - old
        dx = deltas[:, 0]
        dy = deltas[:, 1]

        low, high = 10, 90
        dx_l, dx_h = np.percentile(dx, [low, high])
        dy_l, dy_h = np.percentile(dy, [low, high])
        keep = (dx >= dx_l) & (dx <= dx_h) & (dy >= dy_l) & (dy <= dy_h)
        if np.count_nonzero(keep) >= 5:
            dx = dx[keep]
            dy = dy[keep]

        cam_dx = float(np.median(dx))
        cam_dy = float(np.median(dy))

        self._prev_gray = gray
        self._prev_points = new.reshape(-1, 1, 2)
        if len(self._prev_points) < Settings.MOTION_COMP_MIN_FEATURES // 2:
            self._prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=Settings.MOTION_COMP_MAX_CORNERS,
                qualityLevel=Settings.MOTION_COMP_QUALITY_LEVEL,
                minDistance=Settings.MOTION_COMP_MIN_DISTANCE,
            )
            if self._prev_points is None:
                self._prev_points = None  # Açık atama, sonraki frame L210'da yakalar

        return cam_dx, cam_dy
