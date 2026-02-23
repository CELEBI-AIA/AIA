"""
TEKNOFEST Havacılıkta Yapay Zeka - Ağ İletişim Katmanı (Network Layer)
=======================================================================
Sunucu ile tüm HTTP iletişimini yöneten sınıf.
Retry mekanizması, hata yönetimi ve simülasyon modu desteği içerir.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from config.settings import Settings
from src.utils import Logger, log_json_to_disk


class FrameFetchStatus(str, Enum):
    OK = "ok"
    END_OF_STREAM = "end_of_stream"
    TRANSIENT_ERROR = "transient_error"
    FATAL_ERROR = "fatal_error"


@dataclass
class FrameFetchResult:
    status: FrameFetchStatus
    frame_data: Optional[Dict[str, Any]] = None
    error_type: str = ""
    http_status: Optional[int] = None


class NetworkManager:
    """Sunucu ile iletişimi yöneten ana sınıf."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        simulation_mode: Optional[bool] = None,
    ) -> None:
        self.base_url = base_url or Settings.BASE_URL
        self.simulation_mode = (
            Settings.SIMULATION_MODE if simulation_mode is None else simulation_mode
        )
        self.log = Logger("Network")
        self.session = requests.Session()
        self._frame_counter: int = 0
        self._result_counter: int = 0
        self._sim_image_cache: Optional[np.ndarray] = None

    def start_session(self) -> bool:
        """Sunucu ile oturum başlatır."""
        if self.simulation_mode:
            self.log.success(f"[SIMULATION] Session started -> {self.base_url}")
            return True

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                self.log.info(
                    f"Connecting server... (Attempt {attempt}/{Settings.MAX_RETRIES})"
                )
                response = self.session.get(
                    self.base_url, timeout=Settings.REQUEST_TIMEOUT
                )
                if response.status_code == 200:
                    self.log.success(f"Server connection successful -> {self.base_url}")
                    return True
                self.log.warn(f"Unexpected server response: {response.status_code}")
            except requests.ConnectionError:
                self.log.error(
                    f"Connection error! Retrying in {Settings.RETRY_DELAY}s..."
                )
            except requests.Timeout:
                self.log.error(
                    f"Connection timeout! Retrying in {Settings.RETRY_DELAY}s..."
                )
            except Exception as exc:
                self.log.error(f"Unexpected startup error: {exc}")
            time.sleep(Settings.RETRY_DELAY)

        self.log.error("Server unavailable after all retries.")
        return False

    def get_frame(self) -> FrameFetchResult:
        """Sunucudan bir sonraki video karesinin meta verisini çeker."""
        if self.simulation_mode:
            return FrameFetchResult(
                status=FrameFetchStatus.OK,
                frame_data=self._get_simulation_frame(),
            )

        url = f"{self.base_url}{Settings.ENDPOINT_NEXT_FRAME}"

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.get(url, timeout=Settings.REQUEST_TIMEOUT)

                if response.status_code == 200:
                    data = response.json()
                    if not self._validate_frame_data(data):
                        self.log.error("Invalid frame schema from server.")
                        return FrameFetchResult(
                            status=FrameFetchStatus.FATAL_ERROR,
                            error_type="invalid_frame_schema",
                            http_status=200,
                        )

                    if self._should_log_json(self._frame_counter):
                        log_json_to_disk(
                            data,
                            direction="incoming",
                            tag=f"frame_{self._frame_counter}",
                        )
                    self._frame_counter += 1
                    return FrameFetchResult(
                        status=FrameFetchStatus.OK,
                        frame_data=data,
                        http_status=200,
                    )

                if response.status_code == 204:
                    self.log.info("Video finished (204 No Content)")
                    return FrameFetchResult(
                        status=FrameFetchStatus.END_OF_STREAM,
                        http_status=204,
                    )

                if 500 <= response.status_code < 600:
                    self.log.warn(
                        f"Server temporary error: HTTP {response.status_code}"
                    )
                    time.sleep(Settings.RETRY_DELAY)
                    continue

                self.log.error(f"Unexpected frame response: HTTP {response.status_code}")
                return FrameFetchResult(
                    status=FrameFetchStatus.FATAL_ERROR,
                    error_type="unexpected_http",
                    http_status=response.status_code,
                )

            except (requests.ConnectionError, requests.Timeout) as exc:
                self.log.warn(
                    f"Frame fetch transient error ({type(exc).__name__}) "
                    f"Attempt {attempt}/{Settings.MAX_RETRIES}"
                )
            except ValueError as exc:
                self.log.error(f"JSON parse error: {exc}")
                return FrameFetchResult(
                    status=FrameFetchStatus.FATAL_ERROR,
                    error_type="json_parse",
                )
            except Exception as exc:
                self.log.warn(f"Frame fetch transient exception: {exc}")

            time.sleep(Settings.RETRY_DELAY)

        return FrameFetchResult(
            status=FrameFetchStatus.TRANSIENT_ERROR,
            error_type="retries_exhausted",
        )

    def download_image(self, frame_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Sunucudan veya yerel diskten görüntüyü indirir."""
        if self.simulation_mode:
            return self._load_simulation_image()

        frame_url = frame_data.get("frame_url", "") or frame_data.get("image_url", "")
        if not frame_url:
            self.log.error("Frame URL is missing in frame metadata")
            return None

        full_url = frame_url if str(frame_url).startswith("http") else f"{self.base_url}{frame_url}"

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.get(full_url, timeout=Settings.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        self.log.error("Image decode failed")
                        return None
                    self.log.debug(
                        f"Image downloaded: {frame.shape[1]}x{frame.shape[0]}"
                    )
                    return frame

                self.log.warn(f"Image download HTTP {response.status_code}")
            except requests.Timeout:
                self.log.warn(
                    f"Image download timeout attempt {attempt}/{Settings.MAX_RETRIES}"
                )
            except Exception as exc:
                self.log.warn(f"Image download transient error: {exc}")

            time.sleep(Settings.RETRY_DELAY)

        self.log.error("Image download failed after all retries")
        return None

    def send_result(
        self,
        frame_id: Any,
        detected_objects: List[Dict],
        detected_translation: Dict[str, float],
        frame_data: Optional[Dict[str, Any]] = None,
        frame_shape: Optional[tuple] = None,
    ) -> bool:
        """Tespit ve konum sonuçlarını TEKNOFEST taslak şemasına uyumlu JSON ile gönderir."""
        payload = self.build_competition_payload(
            frame_id=frame_id,
            detected_objects=detected_objects,
            detected_translation=detected_translation,
            frame_data=frame_data,
            frame_shape=frame_shape,
        )

        if self._should_log_json(self._result_counter):
            log_json_to_disk(payload, direction="outgoing", tag=f"result_{frame_id}")
        self._result_counter += 1

        if self.simulation_mode:
            self.log.success(
                f"[SIMULATION] Result prepared -> Frame: {frame_id} | "
                f"Objects: {len(payload['detected_objects'])}"
            )
            return True

        url = f"{self.base_url}{Settings.ENDPOINT_SUBMIT_RESULT}"

        for attempt in range(1, Settings.MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=Settings.REQUEST_TIMEOUT,
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code == 200:
                    self.log.debug(f"Result sent successfully: Frame {frame_id}")
                    return True

                self.log.warn(
                    f"Submit response HTTP {response.status_code} "
                    f"(attempt {attempt}/{Settings.MAX_RETRIES})"
                )
            except requests.Timeout:
                self.log.warn(
                    f"Submit timeout attempt {attempt}/{Settings.MAX_RETRIES}"
                )
            except Exception as exc:
                self.log.warn(
                    f"Submit transient error ({type(exc).__name__}): {exc} "
                    f"(attempt {attempt}/{Settings.MAX_RETRIES})"
                )

            time.sleep(Settings.RETRY_DELAY)

        self.log.error(f"Result submission failed after retries for frame {frame_id}")
        return False

    @staticmethod
    def build_competition_payload(
        frame_id: Any,
        detected_objects: List[Dict],
        detected_translation: Dict[str, float],
        frame_data: Optional[Dict[str, Any]] = None,
        frame_shape: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """TEKNOFEST taslak şemasına uyumlu payload builder."""
        frame_data = frame_data or {}
        frame_h = frame_w = None
        if frame_shape and len(frame_shape) >= 2:
            frame_h = int(frame_shape[0])
            frame_w = int(frame_shape[1])

        clean_objects: List[Dict[str, Any]] = []
        for obj in detected_objects:
            cls = str(obj.get("cls", ""))
            landing = str(obj.get("landing_status", "-1"))
            motion = str(obj.get("motion_status", obj.get("movement_status", "-1")))
            if cls not in {"0", "1", "2", "3"}:
                continue
            if landing not in {"-1", "0", "1"}:
                landing = "-1"
            if motion not in {"-1", "0", "1"}:
                motion = "-1"

            x1 = NetworkManager._safe_int(obj.get("top_left_x", 0))
            y1 = NetworkManager._safe_int(obj.get("top_left_y", 0))
            x2 = NetworkManager._safe_int(obj.get("bottom_right_x", 0))
            y2 = NetworkManager._safe_int(obj.get("bottom_right_y", 0))

            if frame_w is not None and frame_h is not None:
                x1, y1, x2, y2 = NetworkManager._clamp_bbox(
                    x1, y1, x2, y2, frame_w=frame_w, frame_h=frame_h
                )

            clean_objects.append(
                {
                    "cls": cls,
                    "landing_status": landing,
                    "motion_status": motion,
                    "top_left_x": x1,
                    "top_left_y": y1,
                    "bottom_right_x": x2,
                    "bottom_right_y": y2,
                }
            )

        tx = NetworkManager._safe_float(detected_translation.get("translation_x", 0.0))
        ty = NetworkManager._safe_float(detected_translation.get("translation_y", 0.0))
        tz = NetworkManager._safe_float(detected_translation.get("translation_z", 0.0))

        payload_id = frame_data.get("id", frame_id)
        payload_user = frame_data.get("user", Settings.TEAM_NAME)
        payload_frame = frame_data.get("url", frame_data.get("frame", frame_id))

        return {
            "id": payload_id,
            "user": payload_user,
            "frame": payload_frame,
            "detected_objects": clean_objects,
            "detected_translations": [
                {
                    "translation_x": tx,
                    "translation_y": ty,
                    "translation_z": tz,
                }
            ],
            "detected_undefined_objects": [],
        }

    def _get_simulation_frame(self) -> Dict[str, Any]:
        frame_id = self._frame_counter
        self._frame_counter += 1
        return {
            "id": frame_id,
            "url": f"/simulation/frames/{frame_id}",
            "image_url": Settings.SIMULATION_IMAGE_PATH,
            "session": "/simulation/session/1",
            "frame_url": Settings.SIMULATION_IMAGE_PATH,
            "frame_id": frame_id,
            "video_name": "simulation_video",
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 50.0,
            "gps_health_status": 1,
            "gps_health": 1,
        }

    def _load_simulation_image(self) -> Optional[np.ndarray]:
        if self._sim_image_cache is not None:
            return self._sim_image_cache.copy()

        img_path = Settings.SIMULATION_IMAGE_PATH
        frame = cv2.imread(img_path)
        if frame is None:
            self.log.error(f"Simulation image failed: {img_path}")
            return None

        self._sim_image_cache = frame
        self.log.debug(f"Simulation image cached: {frame.shape[1]}x{frame.shape[0]}")
        return frame

    def _validate_frame_data(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            self.log.warn("Frame metadata is not dict")
            return False

        # Şartname taslak alanları: id, url, image_url, ... + yarışma günü değişebilecek varyasyonlar.
        # Uyum için frame_id'yi ortak dahili anahtar olarak normalize ediyoruz.
        frame_id = data.get("frame_id")
        if frame_id is None:
            frame_id = data.get("id")
        if frame_id is None:
            frame_id = data.get("url")
        if frame_id is None:
            frame_id = data.get("frame")

        if frame_id is None:
            self.log.warn("Missing required frame identifier (frame_id/id/url/frame)")
            return False
        data["frame_id"] = frame_id

        # Görsel URL alanlarını normalize et
        if not data.get("frame_url") and data.get("image_url"):
            data["frame_url"] = data.get("image_url")
        if not data.get("image_url") and data.get("frame_url"):
            data["image_url"] = data.get("frame_url")

        health_val = data.get("gps_health")
        if health_val is None:
            health_val = data.get("gps_health_status", 0)
        try:
            if health_val is None or str(health_val).strip().lower() in {
                "unknown",
                "none",
                "null",
                "",
            }:
                data["gps_health"] = 0
            else:
                data["gps_health"] = int(float(health_val))
        except (ValueError, TypeError):
            self.log.warn(f"Corrupt gps_health value: {health_val!r}, forcing 0")
            data["gps_health"] = 0
        data["gps_health_status"] = data["gps_health"]

        for key in ["translation_x", "translation_y", "translation_z", "altitude"]:
            if key in data:
                val = data.get(key)
                try:
                    if val is None or str(val).strip().lower() in {
                        "unknown",
                        "none",
                        "null",
                        "",
                    }:
                        data[key] = 0.0
                    else:
                        data[key] = float(val)
                except (ValueError, TypeError):
                    self.log.warn(f"Corrupt {key}: {val!r}, forcing 0.0")
                    data[key] = 0.0

        return True

    @staticmethod
    def _clamp_bbox(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_w: int,
        frame_h: int,
    ) -> tuple:
        max_x = max(frame_w - 1, 0)
        max_y = max(frame_h - 1, 0)

        x1 = max(0, min(x1, max_x))
        y1 = max(0, min(y1, max_y))
        x2 = max(0, min(x2, max_x))
        y2 = max(0, min(y2, max_y))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return x1, y1, x2, y2

    @staticmethod
    def _safe_int(val: Any) -> int:
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _should_log_json(counter: int) -> bool:
        if not Settings.ENABLE_JSON_LOGGING:
            return False
        interval = max(1, int(Settings.JSON_LOG_EVERY_N_FRAMES))
        return counter % interval == 0
