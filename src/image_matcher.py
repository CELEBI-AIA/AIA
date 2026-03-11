"""Task 3 reference-object matching (ORB/SIFT) with robust input validation."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings
from src.task3_reference_policy import canonicalize_task3_references
from src.utils import Logger


class ReferenceObject:
    """Reference-object feature bundle."""

    def __init__(
        self,
        object_id: int,
        image: np.ndarray,
        keypoints: list,
        descriptors: Optional[np.ndarray],
        fallback_keypoints: Optional[list] = None,
        fallback_descriptors: Optional[np.ndarray] = None,
        label: str = "",
    ) -> None:
        self.object_id = object_id
        self.image = image
        self.h, self.w = image.shape[:2]
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.fallback_keypoints = fallback_keypoints or []
        self.fallback_descriptors = fallback_descriptors
        self.label = label


class ImageMatcher:
    """Reference-object matching via feature extraction and homography."""

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
            "dropped_by_cap": 0,
            "batch_count": 0,
        }
        self._frame_counter: int = 0

        method = str(Settings.TASK3_FEATURE_METHOD).upper()
        self.detector, self.norm_type, self._primary_method = self._build_feature_backend(
            method
        )
        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)

        self._domain_fallback_enabled = bool(
            getattr(Settings, "TASK3_DOMAIN_FALLBACK_ENABLED", True)
        )
        self._domain_fallback_interval = max(
            1, int(getattr(Settings, "TASK3_DOMAIN_FALLBACK_INTERVAL", 3))
        )
        self._domain_fallback_threshold = float(
            getattr(Settings, "TASK3_DOMAIN_FALLBACK_THRESHOLD", 0.58)
        )
        self._fallback_detector = None
        self._fallback_matcher = None
        self._fallback_method = ""
        if self._domain_fallback_enabled:
            fallback_method_raw = str(
                getattr(Settings, "TASK3_DOMAIN_FALLBACK_METHOD", "AKAZE")
            ).upper()
            detector, norm_type, resolved = self._build_feature_backend(
                fallback_method_raw
            )
            if resolved != self._primary_method:
                self._fallback_detector = detector
                self._fallback_matcher = cv2.BFMatcher(norm_type, crossCheck=False)
                self._fallback_method = resolved
            else:
                self._domain_fallback_enabled = False

        self.log.info(f"Feature method: {self._primary_method}")
        if self._domain_fallback_enabled:
            self.log.info(
                "Task3 domain fallback descriptor: "
                f"{self._fallback_method} (interval={self._domain_fallback_interval})"
            )
        self.log.info(
            f"ImageMatcher initialized | similarity_threshold={Settings.TASK3_SIMILARITY_THRESHOLD} "
            f"| fallback_threshold={Settings.TASK3_FALLBACK_THRESHOLD}"
        )

    def _build_feature_backend(self, method: str) -> Tuple[Any, int, str]:
        normalized = str(method or "ORB").strip().upper()
        if normalized == "SIFT" and hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(), cv2.NORM_L2, "SIFT"
        if normalized == "AKAZE":
            return cv2.AKAZE_create(), cv2.NORM_HAMMING, "AKAZE"
        if normalized == "BRISK":
            return cv2.BRISK_create(), cv2.NORM_HAMMING, "BRISK"
        return cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING, "ORB"

    @staticmethod
    def _extract_features(image_gray: np.ndarray, detector: Any) -> Tuple[list, Optional[np.ndarray]]:
        if detector is None:
            return [], None
        keypoints, descriptors = detector.detectAndCompute(image_gray, None)
        keypoints = keypoints or []
        return keypoints, descriptors

    @staticmethod
    def _reference_priority(record: Any, index: int) -> Tuple[float, float, int]:
        if not isinstance(record, dict):
            return (-1.0, -1.0, index)

        try:
            explicit_priority = float(record.get("priority", 0.0))
        except (TypeError, ValueError):
            explicit_priority = 0.0

        source_score = 0.0
        if record.get("image") is not None:
            source_score = 3.0
        elif record.get("path"):
            source_score = 2.0
        elif record.get("image_base64"):
            source_score = 1.0
        return (-explicit_priority, -source_score, index)

    def _prioritize_references(
        self,
        reference_images: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        max_refs = max(1, int(getattr(Settings, "TASK3_MAX_REFERENCES", 1)))
        batch_size = max(1, int(getattr(Settings, "TASK3_REFERENCE_BATCH_SIZE", 1)))

        ranked = sorted(
            enumerate(reference_images),
            key=lambda item: self._reference_priority(item[1], item[0]),
        )
        selected = [item[1] for item in ranked[:max_refs]]
        dropped = max(0, len(reference_images) - len(selected))
        batch_count = (len(selected) + batch_size - 1) // batch_size if selected else 0
        return selected, {
            "max_refs": max_refs,
            "batch_size": batch_size,
            "batch_count": batch_count,
            "dropped_by_cap": dropped,
        }

    def load_references(self, reference_images: List[Dict[str, Any]]) -> int:
        self.references.clear()
        self._references_by_id.clear()
        self._reference_lifecycle.clear()

        selected_references, cap_meta = self._prioritize_references(reference_images)
        canonical_refs, canonical_stats, _, _, _ = canonicalize_task3_references(
            log=self.log,
            references=selected_references,
        )
        self._last_load_stats = {
            "total": len(reference_images),
            "valid": 0,
            "duplicate": int(canonical_stats.get("duplicate", 0)),
            "quarantined": int(canonical_stats.get("quarantined", 0)),
            "dropped_by_cap": int(cap_meta["dropped_by_cap"]),
            "batch_count": 0,
        }

        if cap_meta["dropped_by_cap"] > 0:
            self.log.warn(
                "event=task3_ref_cap_applied "
                f"total={len(reference_images)} selected={len(selected_references)} "
                f"dropped={cap_meta['dropped_by_cap']} max_refs={cap_meta['max_refs']}"
            )

        batch_size = max(1, int(cap_meta["batch_size"]))
        effective_batch_count = (
            (len(canonical_refs) + batch_size - 1) // batch_size
            if canonical_refs
            else 0
        )
        self._last_load_stats["batch_count"] = effective_batch_count
        loaded = 0

        for batch_index in range(effective_batch_count):
            start = batch_index * batch_size
            end = min(len(canonical_refs), start + batch_size)
            batch_refs = canonical_refs[start:end]
            self.log.info(
                "event=task3_ref_batch_load "
                f"batch_index={batch_index + 1}/{effective_batch_count} "
                f"batch_size={len(batch_refs)}"
            )

            for idx, ref_data in enumerate(batch_refs, start=start):
                object_id = int(ref_data.get("object_id", -1))
                self._reference_lifecycle[object_id] = "received"

                label = ref_data.get("label", f"ref_{object_id}")
                if "image" in ref_data:
                    image = ref_data.get("image")
                    if image is None:
                        self._last_load_stats["quarantined"] += 1
                        self.log.warn(
                            f"event=task3_ref_quarantined reason=image_none object_id={object_id} index={idx}"
                        )
                        continue
                    if not isinstance(image, np.ndarray):
                        self._last_load_stats["quarantined"] += 1
                        self.log.warn(
                            f"event=task3_ref_quarantined reason=invalid_image_type object_id={object_id} index={idx}"
                        )
                        continue
                    if image.ndim not in (2, 3):
                        self._last_load_stats["quarantined"] += 1
                        self.log.warn(
                            f"event=task3_ref_quarantined reason=invalid_image_ndim object_id={object_id} index={idx} ndim={image.ndim}"
                        )
                        continue
                elif "path" in ref_data and os.path.isfile(ref_data["path"]):
                    image = cv2.imread(ref_data["path"])
                    if image is None:
                        self._last_load_stats["quarantined"] += 1
                        self.log.warn(f"Reference unreadable: {ref_data['path']}")
                        continue
                else:
                    self._last_load_stats["quarantined"] += 1
                    self.log.warn(f"Reference #{object_id} has no valid image source")
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                keypoints, descriptors = self._extract_features(gray, self.detector)
                fallback_keypoints: list = []
                fallback_descriptors = None
                if self._domain_fallback_enabled and self._fallback_detector is not None:
                    fallback_keypoints, fallback_descriptors = self._extract_features(
                        gray,
                        self._fallback_detector,
                    )

                primary_ready = descriptors is not None and len(keypoints) >= 4
                fallback_ready = (
                    fallback_descriptors is not None and len(fallback_keypoints) >= 4
                )
                if not primary_ready and not fallback_ready:
                    self._last_load_stats["quarantined"] += 1
                    self.log.warn(
                        f"Reference #{object_id}: insufficient features "
                        f"(primary={len(keypoints) if keypoints else 0}, "
                        f"fallback={len(fallback_keypoints) if fallback_keypoints else 0})"
                    )
                    continue

                self._reference_lifecycle[object_id] = "validated"
                ref_obj = ReferenceObject(
                    object_id=object_id,
                    image=image,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    fallback_keypoints=fallback_keypoints,
                    fallback_descriptors=fallback_descriptors,
                    label=label,
                )
                self.references.append(ref_obj)
                self._references_by_id[object_id] = ref_obj
                self._reference_lifecycle[object_id] = "loaded"
                self._last_load_stats["valid"] += 1
                loaded += 1
                self.log.info(
                    f"Reference #{object_id} loaded: {ref_obj.w}x{ref_obj.h}px, "
                    f"primary={len(keypoints)} fallback={len(fallback_keypoints)}"
                )

        self.references = list(self._references_by_id.values())
        self.log.info(
            f"event=task3_ref_validation_summary total={self._last_load_stats['total']} "
            f"valid={self._last_load_stats['valid']} duplicate={self._last_load_stats['duplicate']} "
            f"quarantined={self._last_load_stats['quarantined']}"
        )
        self.log.success(f"Total loaded references: {loaded}/{len(reference_images)}")
        return loaded

    def load_references_from_directory(self, directory: Optional[str] = None) -> int:
        ref_dir = directory or Settings.TASK3_REFERENCE_DIR
        if not os.path.isdir(ref_dir):
            self.log.warn(f"Reference directory not found: {ref_dir}")
            return 0

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ref_list: List[Dict[str, Any]] = []
        files = sorted(os.listdir(ref_dir))
        for idx, fname in enumerate(files, start=1):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in image_exts:
                continue
            ref_list.append(
                {
                    "object_id": idx,
                    "path": os.path.join(ref_dir, fname),
                    "label": os.path.splitext(fname)[0],
                }
            )
        return self.load_references(ref_list)

    def match(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        self._frame_counter += 1
        if not self.references:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_kp, frame_desc = self._extract_features(gray, self.detector)

        should_try_domain_fallback = (
            self._domain_fallback_enabled
            and self._fallback_detector is not None
            and self._frame_counter % self._domain_fallback_interval == 0
        )
        fallback_frame_kp: list = []
        fallback_frame_desc = None
        if should_try_domain_fallback:
            fallback_frame_kp, fallback_frame_desc = self._extract_features(
                gray,
                self._fallback_detector,
            )

        if (
            (frame_desc is None or len(frame_kp) < 4)
            and (fallback_frame_desc is None or len(fallback_frame_kp) < 4)
        ):
            return []

        results: List[Dict[str, Any]] = []
        for ref in self.references:
            match_result = self._match_reference(
                ref,
                frame_kp,
                frame_desc,
                gray.shape,
                fallback_frame_kp=fallback_frame_kp,
                fallback_frame_desc=fallback_frame_desc,
            )
            parsed = self._parse_match_result(match_result)
            if parsed is None:
                continue

            x1, y1, x2, y2, quality_score, quality_flag = parsed
            if ref.object_id not in self._references_by_id:
                continue

            self._reference_lifecycle[ref.object_id] = "matched"
            result_obj: Dict[str, Any] = {
                "object_id": ref.object_id,
                "top_left_x": x1,
                "top_left_y": y1,
                "bottom_right_x": x2,
                "bottom_right_y": y2,
            }
            if bool(getattr(Settings, "TASK3_INCLUDE_QUALITY_FIELDS", False)):
                result_obj["quality_score"] = round(max(0.0, min(1.0, quality_score)), 4)
                result_obj["quality_flag"] = quality_flag
            results.append(result_obj)
            if isinstance(match_result, dict) and bool(
                match_result.get("used_fallback_descriptor", False)
            ):
                self.log.info(
                    f"event=task3_domain_fallback_used object_id={ref.object_id} "
                    f"method={self._fallback_method} score={quality_score:.4f}"
                )
            self.log.debug(
                "event=task3_match_quality "
                f"object_id={ref.object_id} score={quality_score:.4f} flag={quality_flag}"
            )

        if results:
            self.log.debug(f"Frame {self._frame_counter}: matched references={len(results)}")
        return results

    @staticmethod
    def _parse_match_result(match_result: Any) -> Optional[Tuple[float, float, float, float, float, str]]:
        if match_result is None:
            return None

        if isinstance(match_result, dict):
            bbox = match_result.get("bbox")
            if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
                return None
            try:
                x1, y1, x2, y2 = (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
                score = float(match_result.get("quality_score", 0.0))
            except (TypeError, ValueError):
                return None
            flag = str(match_result.get("quality_flag", "unknown"))
            return x1, y1, x2, y2, score, flag

        if isinstance(match_result, (tuple, list)) and len(match_result) >= 4:
            try:
                x1, y1, x2, y2 = (
                    float(match_result[0]),
                    float(match_result[1]),
                    float(match_result[2]),
                    float(match_result[3]),
                )
            except (TypeError, ValueError):
                return None
            return x1, y1, x2, y2, 0.0, "unknown"

        return None

    def _match_reference(
        self,
        ref: ReferenceObject,
        frame_kp: list,
        frame_desc: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
        fallback_frame_kp: Optional[list] = None,
        fallback_frame_desc: Optional[np.ndarray] = None,
    ) -> Optional[Any]:
        threshold = float(Settings.TASK3_SIMILARITY_THRESHOLD)
        if self._frame_counter % max(1, int(Settings.TASK3_FALLBACK_INTERVAL)) == 0:
            threshold = float(Settings.TASK3_FALLBACK_THRESHOLD)

        primary = self._match_with_features(
            ref_keypoints=ref.keypoints,
            ref_descriptors=ref.descriptors,
            frame_kp=frame_kp,
            frame_desc=frame_desc,
            frame_shape=frame_shape,
            matcher=self.matcher,
            threshold=threshold,
        )
        if primary is not None:
            quality = float(primary.get("quality_score", 0.0))
            primary["quality_flag"] = self._quality_flag(quality)
            primary["used_fallback_descriptor"] = False
            return primary

        can_try_domain_fallback = (
            self._domain_fallback_enabled
            and self._fallback_detector is not None
            and self._fallback_matcher is not None
            and self._frame_counter % self._domain_fallback_interval == 0
        )
        if not can_try_domain_fallback:
            return None

        fallback = self._match_with_features(
            ref_keypoints=ref.fallback_keypoints,
            ref_descriptors=ref.fallback_descriptors,
            frame_kp=fallback_frame_kp or [],
            frame_desc=fallback_frame_desc,
            frame_shape=frame_shape,
            matcher=self._fallback_matcher,
            threshold=self._domain_fallback_threshold,
        )
        if fallback is None:
            return None

        quality = float(fallback.get("quality_score", 0.0))
        fallback["quality_flag"] = f"{self._quality_flag(quality)}_domain_fallback"
        fallback["used_fallback_descriptor"] = True
        return fallback

    @staticmethod
    def _match_with_features(
        ref_keypoints: list,
        ref_descriptors: Optional[np.ndarray],
        frame_kp: list,
        frame_desc: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
        matcher: Any,
        threshold: float,
    ) -> Optional[Dict[str, Any]]:
        if (
            ref_descriptors is None
            or frame_desc is None
            or len(ref_keypoints) < 4
            or len(frame_kp) < 4
            or matcher is None
        ):
            return None

        try:
            matches = matcher.knnMatch(ref_descriptors, frame_desc, k=2)
        except cv2.error:
            return None

        good_matches = []
        for m_pair in matches:
            if len(m_pair) < 2:
                continue
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        min_matches = max(4, int(len(ref_keypoints) * 0.05))
        if len(good_matches) < min_matches:
            return None

        similarity = len(good_matches) / max(1, len(ref_keypoints))
        if similarity < float(threshold):
            return None

        try:
            src_pts = np.float32(
                [ref_keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            if len(np.unique(src_pts.reshape(-1, 2), axis=0)) < 4:
                return None
            if len(np.unique(dst_pts.reshape(-1, 2), axis=0)) < 4:
                return None

            mat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mat is None or mat.shape != (3, 3):
                pts = dst_pts.reshape(-1, 2)
            else:
                src_h = max(2.0, float(src_pts[:, 0, 1].max() - src_pts[:, 0, 1].min()))
                src_w = max(2.0, float(src_pts[:, 0, 0].max() - src_pts[:, 0, 0].min()))
                corners = np.float32([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]]).reshape(
                    -1, 1, 2
                )
                transformed = cv2.perspectiveTransform(corners, mat)
                pts = transformed.reshape(-1, 2)
                if len(pts) >= 4:
                    hull = cv2.convexHull(pts.astype(np.float32))
                    if hull is None or len(hull) < 4 or not cv2.isContourConvex(hull):
                        return None

            x1 = float(max(0, pts[:, 0].min()))
            y1 = float(max(0, pts[:, 1].min()))
            x2 = float(
                min(
                    frame_shape[1] if len(frame_shape) > 1 else frame_shape[0],
                    pts[:, 0].max(),
                )
            )
            y2 = float(min(frame_shape[0], pts[:, 1].max()))
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < 5 or bbox_h < 5:
                return None
            if bbox_w > frame_shape[1] * 0.8 or bbox_h > frame_shape[0] * 0.8:
                return None

            return {
                "bbox": (x1, y1, x2, y2),
                "quality_score": similarity,
            }
        except (cv2.error, ValueError, IndexError):
            return None

    @staticmethod
    def _quality_flag(similarity: float) -> str:
        high = float(getattr(Settings, "TASK3_QUALITY_HIGH_THRESHOLD", 0.85))
        medium = float(getattr(Settings, "TASK3_QUALITY_MEDIUM_THRESHOLD", 0.72))
        score = float(similarity)
        if score >= high:
            return "high"
        if score >= medium:
            return "medium"
        return "low"

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
            "dropped_by_cap": 0,
            "batch_count": 0,
        }
        self._frame_counter = 0
        self.log.info("ImageMatcher reset")

    @property
    def id_lifecycle_states(self) -> Dict[int, str]:
        return dict(self._reference_lifecycle)

    @property
    def last_load_stats(self) -> Dict[str, int]:
        return dict(self._last_load_stats)
