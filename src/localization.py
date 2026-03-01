"""Hibrit konum kestirimi: GPS=1 ise sunucu verisi, GPS=0 ise Lucas-Kanade optik akış.
Piksel kayması focal_length ve irtifa ile metreye çevrilir."""

from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.frame_context import FrameContext

import cv2
import numpy as np

from config.settings import Settings
from src.utils import Logger


class VisualOdometry:
    """GPS + optik akış hibrit pozisyon kestirimi."""

    def __init__(self) -> None:
        self.log = Logger("Localization")

        self.position: Dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }

        self._last_gps_position: Optional[Dict[str, float]] = None

        self._last_gps_altitude: float = Settings.DEFAULT_ALTITUDE

        self._ema_alpha: float = 0.4
        self._ema_dx: float = 0.0
        self._ema_dy: float = 0.0

        self._max_displacement_per_frame: float = 5.0

        self._was_gps_healthy: bool = False

        self._last_of_position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._initial_point_count: int = 0

        self._feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=40,
            blockSize=7,
        )
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        self.log.info("Visual Odometry başlatıldı — Başlangıç: (0, 0, 0)")
        if Settings.FOCAL_LENGTH_PX == 800.0:
            self.log.warn(
                "FOCAL_LENGTH_PX=800 (varsayılan) kullanılıyor. "
                "TBD-010 kamera parametreleri yayımlandığında config/settings.py güncellenmeli."
            )

    def update(
        self,
        frame_ctx: "FrameContext",
        server_data: Dict,
    ) -> Dict[str, float]:
        gps_health = server_data.get("gps_health", 0)

        if isinstance(frame_ctx, np.ndarray):
            gray = cv2.cvtColor(frame_ctx, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_ctx.gray

        if gps_health == 1:
            # GPS sağlıklı: sunucu verisini kullan, gri kareyi referans için sakla
            self._update_from_gps(server_data)
            self._prev_gray = gray
            self._was_gps_healthy = True

        else:
            # GPS kapalı: optik akış. İlk geçişte referans kare oluştur
            if self._was_gps_healthy:
                self._update_reference_frame(
                    self._prev_gray if self._prev_gray is not None else gray
                )
                self._was_gps_healthy = False
                self._ema_dx = 0.0
                self._ema_dy = 0.0

                self.log.info("GPS → Optik Akış geçişi — referans kare oluşturuldu, EMA resetlendi.")

            if self._prev_gray is not None and self._prev_points is not None:
                self._update_from_optical_flow(gray, server_data)
            else:
                self.log.warn(
                    "GPS mevcut değil ve henüz referans kare oluşmadı — "
                    "GPS yok, referans kare henüz oluşmadı — pozisyon (0,0,0) korunuyor."
                )
                self._update_reference_frame(gray)

        return self.get_position()

    def _update_from_gps(self, server_data: Dict) -> None:
        new_x = float(server_data.get("translation_x", self.position["x"]))
        new_y = float(server_data.get("translation_y", self.position["y"]))
        new_z = float(server_data.get("translation_z", self.position["z"]))

        self.position["x"] = new_x
        self.position["y"] = new_y
        self.position["z"] = new_z

        if new_z > 0:
            self._last_gps_altitude = new_z

        self._last_gps_position = {
            "x": new_x,
            "y": new_y,
            "z": new_z,
        }

        self.log.debug(
            f"GPS güncelleme → X:{new_x:.2f}m Y:{new_y:.2f}m Z:{new_z:.2f}m"
        )

    def _update_from_optical_flow(
        self,
        gray: np.ndarray,
        server_data: Dict,
    ) -> None:
        if self._prev_points is None or len(self._prev_points) < 10:
            self._update_reference_frame(gray)
            self.log.warn("Yetersiz köşe noktası — yeniden tespit ediliyor")
            return

        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        if next_points is None or status is None:
            self._update_reference_frame(gray)
            self.log.warn("Optik Akış başarısız — referans kare yenileniyor")
            return

        mask = status.flatten() == 1

        if self._prev_points is None:
            return

        good_old = self._prev_points[mask].reshape(-1, 2)
        good_new = next_points[mask].reshape(-1, 2)

        if len(good_new) < 5:
            self._update_reference_frame(gray)
            self.log.warn("Başarılı takip sayısı az — referans yenileniyor")
            return

        dx_pixels = float(np.median(good_new[:, 0] - good_old[:, 0]))
        dy_pixels = float(np.median(good_new[:, 1] - good_old[:, 1]))

        scale_ratio = 1.0
        if len(good_new) >= 3:
            good_old_pts = good_old[:, np.newaxis, :]
            good_new_pts = good_new[:, np.newaxis, :]
            dist_old = np.sqrt(np.sum((good_old_pts - good_old[np.newaxis, :, :]) ** 2, axis=-1))
            dist_new = np.sqrt(np.sum((good_new_pts - good_new[np.newaxis, :, :]) ** 2, axis=-1))
            iu = np.triu_indices(len(good_old), k=1)
            old_vals = dist_old[iu]
            new_vals = dist_new[iu]
            valid = old_vals > 5.0
            if np.any(valid):
                scale_ratio = float(np.median(new_vals[valid] / old_vals[valid]))

        raw_alt = server_data.get("translation_z", None)
        try:
            altitude = float(raw_alt)
            if np.isnan(altitude):
                altitude = 0.0
        except (TypeError, ValueError):
            altitude = 0.0

        if altitude <= 0:
            altitude = self._last_gps_altitude

        dx_meters, dy_meters = self._pixel_to_meter(dx_pixels, dy_pixels, altitude)
        alpha = self._ema_alpha
        self._ema_dx = alpha * dx_meters + (1 - alpha) * self._ema_dx
        self._ema_dy = alpha * dy_meters + (1 - alpha) * self._ema_dy

        cap = self._max_displacement_per_frame
        smooth_dx = max(-cap, min(cap, self._ema_dx))
        smooth_dy = max(-cap, min(cap, self._ema_dy))

        dz_meters = 0.0
        if 0.5 < scale_ratio < 2.0 and scale_ratio != 1.0:
            dz_meters = self.position["z"] * ((1.0 / scale_ratio) - 1.0)

        smooth_dz = max(-cap, min(cap, dz_meters * alpha))
        self.position["x"] += smooth_dx
        self.position["y"] += smooth_dy
        self.position["z"] += smooth_dz

        self._last_of_position = {k: v for k, v in self.position.items()}

        self.log.debug(
            f"Optik Akış → dX:{dx_meters:.3f}m dY:{dy_meters:.3f}m dZ:{dz_meters:.3f}m | "
            f"Piksel: ({dx_pixels:.1f}, {dy_pixels:.1f}) | Scale: {scale_ratio:.3f} | "
            f"İrtifa: {altitude:.1f}m | "
            f"Takip: {len(good_new)}/{len(self._prev_points)} nokta"
        )

        if (
            self._initial_point_count > 0
            and len(good_new) < self._initial_point_count * 0.5
        ):
            self._update_reference_frame(gray)
            self.log.debug("Köşe noktası kaybı %50 üzeri — referans yenilendi")
        else:
            self._prev_gray = gray.copy()
            self._prev_points = good_new.reshape(-1, 1, 2)

    def _pixel_to_meter(
        self,
        dx_px: float,
        dy_px: float,
        altitude: float,
    ) -> Tuple[float, float]:
        focal = Settings.FOCAL_LENGTH_PX
        if focal <= 0:
            focal = 800.0

        # Pinhole: metre = piksel * irtifa / focal_length
        dx_m = dx_px * altitude / focal
        dy_m = dy_px * altitude / focal

        return dx_m, dy_m

    def _update_reference_frame(self, gray: np.ndarray) -> None:
        self._prev_gray = gray.copy()
        self._prev_points = cv2.goodFeaturesToTrack(
            gray, **self._feature_params
        )

        n_points = len(self._prev_points) if self._prev_points is not None else 0
        self._initial_point_count = n_points
        self.log.debug(f"Referans kare güncellendi — {n_points} köşe noktası")

    def get_position(self) -> Dict[str, float]:
        return {
            "x": round(self.position["x"], 4),
            "y": round(self.position["y"], 4),
            "z": round(self.position["z"], 4),
        }

    def get_last_of_position(self) -> Dict[str, float]:
        return {
            "x": round(self._last_of_position["x"], 4),
            "y": round(self._last_of_position["y"], 4),
            "z": round(self._last_of_position["z"], 4),
        }

    def reset(self) -> None:
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._last_of_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._prev_gray = None
        self._prev_points = None
        self._last_gps_position = None
        self._initial_point_count = 0
        self.log.info("Visual Odometry sıfırlandı → (0, 0, 0)")
