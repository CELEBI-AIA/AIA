"""Görev 3: Referans obje eşleştirme (ORB/SIFT). Referans görüntülerden feature çıkar, karede ara."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings
from src.utils import Logger


class ReferenceObject:
    """Referans objenin feature bilgileri."""

    def __init__(
        self,
        object_id: int,
        image: np.ndarray,
        keypoints: list,
        descriptors: Optional[np.ndarray],
        label: str = "",
    ) -> None:
        self.object_id = object_id
        self.image = image
        self.h, self.w = image.shape[:2]
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.label = label


class ImageMatcher:
    """Referans obje eşleştirme (ORB/SIFT)."""

    def __init__(self) -> None:
        self.log = Logger("Task3")
        self.references: List[ReferenceObject] = []
        self._references_by_id: Dict[int, ReferenceObject] = {}
        self._reference_lifecycle: Dict[int, str] = {}
        self._last_load_stats: Dict[str, int] = {
            "total": 0,
            "valid": 0,
            "duplicate": 0,
            "quarantined": 0,
        }
        self._telemetry_counters: Dict[str, int] = {
            "task3_low_quality_drop": 0,
            "task3_conflict_drop": 0,
        }
        self._frame_counter: int = 0

        method = Settings.TASK3_FEATURE_METHOD.upper()
        if method == "SIFT":
            self.detector = cv2.SIFT_create()
            self.norm_type = cv2.NORM_L2
            self.log.info("Feature method: SIFT (daha robust, yavaş)")
        else:
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.norm_type = cv2.NORM_HAMMING
            self.log.info("Feature method: ORB (hızlı, offline-uyumlu)")

        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)

        self.log.info(
            f"ImageMatcher initialized | "
            f"similarity_threshold={Settings.TASK3_SIMILARITY_THRESHOLD} | "
            f"fallback_threshold={Settings.TASK3_FALLBACK_THRESHOLD}"
        )

    @staticmethod
    def _normalize_object_id(raw_object_id: Any) -> Optional[int]:
        if isinstance(raw_object_id, bool) or raw_object_id is None:
            return None
        if isinstance(raw_object_id, int):
            object_id = raw_object_id
        elif isinstance(raw_object_id, float):
            if not raw_object_id.is_integer():
                return None
            object_id = int(raw_object_id)
        elif isinstance(raw_object_id, str):
            stripped = raw_object_id.strip()
            if not stripped:
                return None
            try:
                object_id = int(stripped)
            except ValueError:
                return None
        else:
            return None
        if object_id < 0:
            return None
        return object_id

    def load_references(self, reference_images: List[Dict[str, Any]]) -> int:
        self.references.clear()
        self._references_by_id.clear()
        self._reference_lifecycle.clear()
        self._last_load_stats = {
            "total": len(reference_images),
            "valid": 0,
            "duplicate": 0,
            "quarantined": 0,
        }
        loaded = 0

        for idx, ref_data in enumerate(reference_images):
            if not isinstance(ref_data, dict):
                self._last_load_stats["quarantined"] += 1
                self.log.warn(
                    f"event=task3_ref_quarantined reason=invalid_record_type index={idx}"
                )
                continue

            object_id = self._normalize_object_id(ref_data.get("object_id"))
            if object_id is None:
                self._last_load_stats["quarantined"] += 1
                self.log.warn(
                    f"event=task3_ref_quarantined reason=invalid_object_id index={idx} raw_object_id={ref_data.get('object_id')}"
                )
                continue

            self._reference_lifecycle[object_id] = "received"

            if object_id in self._references_by_id:
                self._last_load_stats["duplicate"] += 1
                self._last_load_stats["quarantined"] += 1
                self.log.warn(
                    f"event=task3_ref_duplicate_detected object_id={object_id} index={idx}"
                )
                self.log.warn(
                    f"event=task3_ref_quarantined reason=duplicate_object_id object_id={object_id} index={idx}"
                )
                continue

            label = ref_data.get("label", f"ref_{object_id}")

            if "image" in ref_data and ref_data["image"] is not None:
                image = ref_data["image"]
            elif "path" in ref_data and os.path.isfile(ref_data["path"]):
                image = cv2.imread(ref_data["path"])
                if image is None:
                    self._last_load_stats["quarantined"] += 1
                    self.log.warn(f"Referans obje okunamadı: {ref_data['path']}")
                    continue
            else:
                self._last_load_stats["quarantined"] += 1
                self.log.warn(f"Referans obje #{object_id} için geçerli görüntü bulunamadı")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 4:
                self._last_load_stats["quarantined"] += 1
                self.log.warn(
                    f"Referans obje #{object_id}: yetersiz feature ({len(keypoints) if keypoints else 0})"
                )
                continue

            self._reference_lifecycle[object_id] = "validated"
            ref_obj = ReferenceObject(
                object_id=object_id,
                image=image,
                keypoints=keypoints,
                descriptors=descriptors,
                label=label,
            )
            self.references.append(ref_obj)
            self._references_by_id[object_id] = ref_obj
            self._reference_lifecycle[object_id] = "loaded"
            loaded += 1
            self._last_load_stats["valid"] += 1

            self.log.info(
                f"Referans #{object_id} yüklendi: "
                f"{ref_obj.w}x{ref_obj.h}px, {len(keypoints)} feature"
            )

        self.references = list(self._references_by_id.values())
        self.log.info(
            f"event=task3_ref_validation_summary total={self._last_load_stats['total']} "
            f"valid={self._last_load_stats['valid']} duplicate={self._last_load_stats['duplicate']} "
            f"quarantined={self._last_load_stats['quarantined']}"
        )
        self.log.success(f"Toplam {loaded}/{len(reference_images)} referans obje yüklendi")
        return loaded

    def load_references_from_directory(self, directory: Optional[str] = None) -> int:
        ref_dir = directory or Settings.TASK3_REFERENCE_DIR
        if not os.path.isdir(ref_dir):
            self.log.warn(f"Referans dizini bulunamadı: {ref_dir}")
            return 0

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ref_list: List[Dict[str, Any]] = []

        files = sorted(os.listdir(ref_dir))
        for idx, fname in enumerate(files, start=1):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in image_exts:
                continue

            if len(ref_list) >= Settings.TASK3_MAX_REFERENCES:
                self.log.warn(
                    f"Maks referans limiti ({Settings.TASK3_MAX_REFERENCES}) aşıldı, "
                    f"fazla dosyalar atlanıyor"
                )
                break

            ref_list.append({
                "object_id": idx,
                "path": os.path.join(ref_dir, fname),
                "label": os.path.splitext(fname)[0],
            })

        return self.load_references(ref_list)

    def match(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        self._frame_counter += 1

        if not self.references:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_kp, frame_desc = self.detector.detectAndCompute(gray, None)

        if frame_desc is None or len(frame_kp) < 4:
            return []

        candidates: List[Dict[str, Any]] = []

        for ref in self.references:
            match_candidate = self._match_reference(ref, frame_kp, frame_desc, gray.shape)
            if match_candidate is None:
                continue
            if ref.object_id not in self._references_by_id:
                continue
            if isinstance(match_candidate, tuple) and len(match_candidate) == 4:
                x1, y1, x2, y2 = match_candidate
                match_candidate = {
                    "top_left_x": x1,
                    "top_left_y": y1,
                    "bottom_right_x": x2,
                    "bottom_right_y": y2,
                    "quality_score": 1.0,
                    "similarity": 1.0,
                    "inlier_ratio": 1.0,
                }
            if not isinstance(match_candidate, dict):
                continue
            candidate = dict(match_candidate)
            candidate["object_id"] = ref.object_id
            candidates.append(candidate)

        if not candidates:
            return []

        # object_id başına en güçlü adayı tut.
        best_per_id: Dict[int, Dict[str, Any]] = {}
        for candidate in candidates:
            object_id = int(candidate["object_id"])
            prev = best_per_id.get(object_id)
            if prev is None or self._candidate_rank(candidate) < self._candidate_rank(prev):
                best_per_id[object_id] = candidate

        conflict_iou = float(
            max(0.0, min(1.0, getattr(Settings, "TASK3_CONFLICT_IOU_THRESHOLD", 0.35)))
        )
        sorted_candidates = sorted(best_per_id.values(), key=self._candidate_rank)
        filtered: List[Dict[str, Any]] = []
        conflict_drop_count = 0
        for candidate in sorted_candidates:
            conflict = any(
                self._bbox_iou(candidate, kept) >= conflict_iou for kept in filtered
            )
            if conflict:
                conflict_drop_count += 1
                continue
            filtered.append(candidate)

        if conflict_drop_count > 0:
            self._telemetry_counters["task3_conflict_drop"] += conflict_drop_count
            self.log.warn(
                f"event=task3_conflict_drop frame={self._frame_counter} dropped={conflict_drop_count}"
            )

        results: List[Dict[str, Any]] = []
        for candidate in filtered:
            object_id = int(candidate["object_id"])
            self._reference_lifecycle[object_id] = "matched"
            results.append(
                {
                    "object_id": object_id,
                    "top_left_x": candidate["top_left_x"],
                    "top_left_y": candidate["top_left_y"],
                    "bottom_right_x": candidate["bottom_right_x"],
                    "bottom_right_y": candidate["bottom_right_y"],
                }
            )

        if results:
            self.log.debug(
                f"Frame {self._frame_counter}: {len(results)} referans obje tespit edildi"
            )

        return results

    def _match_reference(
        self,
        ref: ReferenceObject,
        frame_kp: list,
        frame_desc: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        if ref.descriptors is None:
            self._telemetry_counters["task3_low_quality_drop"] += 1
            return None

        try:
            matches = self.matcher.knnMatch(ref.descriptors, frame_desc, k=2)
        except cv2.error:
            self._telemetry_counters["task3_low_quality_drop"] += 1
            return None

        good_matches = []
        for m_pair in matches:
            if len(m_pair) < 2:
                continue
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        min_matches = max(4, int(len(ref.keypoints) * 0.05))
        if len(good_matches) < min_matches:
            self._telemetry_counters["task3_low_quality_drop"] += 1
            return None

        similarity = len(good_matches) / max(1, len(ref.keypoints))

        # Periyodik olarak daha düşük eşik (fallback) denemek için
        threshold = Settings.TASK3_SIMILARITY_THRESHOLD
        if self._frame_counter % Settings.TASK3_FALLBACK_INTERVAL == 0:
            threshold = Settings.TASK3_FALLBACK_THRESHOLD

        if similarity < threshold:
            self._telemetry_counters["task3_low_quality_drop"] += 1
            return None

        try:
            src_pts = np.float32(
                [ref.keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [frame_kp[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            if len(np.unique(src_pts.reshape(-1, 2), axis=0)) < 4:
                self._telemetry_counters["task3_low_quality_drop"] += 1
                return None
            if len(np.unique(dst_pts.reshape(-1, 2), axis=0)) < 4:
                self._telemetry_counters["task3_low_quality_drop"] += 1
                return None

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            fallback_used = False
            inlier_ratio = 0.0
            if M is None or M.shape != (3, 3):
                fallback_used = True
                self.log.warn("Homografi dejenere oldu, nokta bazlı bounding rect (fallback) çıkarıldı")
                pts = dst_pts.reshape(-1, 2)
            else:
                if mask is not None and len(mask) > 0:
                    inlier_ratio = float(mask.sum()) / float(len(mask))
                h, w = ref.h, ref.w
                corners = np.float32(
                    [[0, 0], [w, 0], [w, h], [0, h]]
                ).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, M)
                pts = transformed.reshape(-1, 2)
                if len(pts) >= 4:
                    hull = cv2.convexHull(pts.astype(np.float32))
                    if hull is None or len(hull) < 4 or not cv2.isContourConvex(hull):
                        self._telemetry_counters["task3_low_quality_drop"] += 1
                        return None

            x1 = float(max(0, pts[:, 0].min()))
            y1 = float(max(0, pts[:, 1].min()))
            x2 = float(min(frame_shape[1] if len(frame_shape) > 1 else frame_shape[0], pts[:, 0].max()))
            y2 = float(min(frame_shape[0], pts[:, 1].max()))

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < 5 or bbox_h < 5:
                self._telemetry_counters["task3_low_quality_drop"] += 1
                return None
            if bbox_w > frame_shape[1] * 0.8 or bbox_h > frame_shape[0] * 0.8:
                self._telemetry_counters["task3_low_quality_drop"] += 1
                return None

            quality_score = 0.6 * similarity + 0.4 * inlier_ratio
            min_quality = float(getattr(Settings, "TASK3_MIN_QUALITY_SCORE", 0.35))
            fallback_min_quality = float(
                getattr(Settings, "TASK3_FALLBACK_MIN_QUALITY_SCORE", 0.55)
            )
            min_inlier_ratio = float(getattr(Settings, "TASK3_MIN_INLIER_RATIO", 0.25))

            if (not fallback_used) and inlier_ratio < min_inlier_ratio:
                self._telemetry_counters["task3_low_quality_drop"] += 1
                return None
            if fallback_used and quality_score < fallback_min_quality:
                self._telemetry_counters["task3_low_quality_drop"] += 1
                return None
            if quality_score < min_quality:
                self._telemetry_counters["task3_low_quality_drop"] += 1
                return None

            return {
                "top_left_x": x1,
                "top_left_y": y1,
                "bottom_right_x": x2,
                "bottom_right_y": y2,
                "similarity": similarity,
                "inlier_ratio": inlier_ratio,
                "quality_score": quality_score,
            }

        except (cv2.error, ValueError, IndexError):
            self._telemetry_counters["task3_low_quality_drop"] += 1
            return None

    @property
    def reference_count(self) -> int:
        return len(self.references)

    @property
    def is_ready(self) -> bool:
        return len(self.references) > 0

    def reset(self) -> None:
        self.references.clear()
        self._references_by_id.clear()
        self._reference_lifecycle.clear()
        self._last_load_stats = {
            "total": 0,
            "valid": 0,
            "duplicate": 0,
            "quarantined": 0,
        }
        self._telemetry_counters = {
            "task3_low_quality_drop": 0,
            "task3_conflict_drop": 0,
        }
        self._frame_counter = 0
        self.log.info("ImageMatcher reset")

    @property
    def id_lifecycle_states(self) -> Dict[int, str]:
        return dict(self._reference_lifecycle)

    @property
    def last_load_stats(self) -> Dict[str, int]:
        return dict(self._last_load_stats)

    @property
    def telemetry_counters(self) -> Dict[str, int]:
        return dict(self._telemetry_counters)

    @staticmethod
    def _bbox_iou(det_a: Dict[str, Any], det_b: Dict[str, Any]) -> float:
        ax1 = float(det_a.get("top_left_x", 0))
        ay1 = float(det_a.get("top_left_y", 0))
        ax2 = float(det_a.get("bottom_right_x", 0))
        ay2 = float(det_a.get("bottom_right_y", 0))
        bx1 = float(det_b.get("top_left_x", 0))
        by1 = float(det_b.get("top_left_y", 0))
        bx2 = float(det_b.get("bottom_right_x", 0))
        by2 = float(det_b.get("bottom_right_y", 0))
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    @staticmethod
    def _candidate_rank(candidate: Dict[str, Any]) -> Tuple[float, float, int, int]:
        x1 = int(float(candidate.get("top_left_x", 0)))
        y1 = int(float(candidate.get("top_left_y", 0)))
        x2 = int(float(candidate.get("bottom_right_x", 0)))
        y2 = int(float(candidate.get("bottom_right_y", 0)))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        score = float(candidate.get("quality_score", 0.0))
        return (-score, -float(area), x1, y1)
