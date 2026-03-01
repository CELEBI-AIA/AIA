"""
TEKNOFEST Havacılıkta Yapay Zeka - Konum Kestirimi Modülü (Görev 2)
====================================================================
Hibrit Visual Odometry (Görsel Odometri) sistemi.

Mantık:
    - GPS sağlıklı (gps_health=1): Sunucu verisini kullan, sistemi kalibre et.
    - GPS sağlıksız (gps_health=0): Lucas-Kanade Optik Akış ile piksel kaydırma
      hesapla, focal length ve irtifa kullanarak metreye çevir.

Koordinat Sistemi:
    - Referans noktası: İlk kare (x0=0, y0=0, z0=0)
    - Birim: Metre

Kullanım:
    from src.localization import VisualOdometry
    vo = VisualOdometry()
    position = vo.update(frame, server_data)
"""

from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.frame_context import FrameContext

import cv2
import numpy as np

from config.settings import Settings
from src.utils import Logger


class VisualOdometry:
    """
    Hibrit kamera tabanlı pozisyon kestirimi sınıfı.

    GPS sağlıklıyken sunucu verisini kullanır, GPS kesildiğinde
    sparse optical flow (Lucas-Kanade) yöntemiyle kendi konumunu hesaplar.

    Attributes:
        position: Güncel (x, y, z) pozisyonu (metre).
        prev_gray: Bir önceki karenin gri tonlamalı hali.
        prev_points: Bir önceki karede takip edilen köşe noktaları.
    """

    def __init__(self) -> None:
        """
        Visual Odometry başlangıç durumunu ayarlar.

        Başlangıç pozisyonu: (0.0, 0.0, 0.0) — şartname gereği.
        """
        self.log = Logger("Localization")

        # Güncel pozisyon (metre)
        self.position: Dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }

        # En son bilinen GPS pozisyonu (kalibrasyon referansı)
        self._last_gps_position: Optional[Dict[str, float]] = None

        # Son bilinen geçerli GPS irtifası (NaN fallback için)
        self._last_gps_altitude: float = Settings.DEFAULT_ALTITUDE

        # EMA yumuşatma katsayısı (0.0–1.0, düşük = daha fazla yumuşatma)
        self._ema_alpha: float = 0.4
        self._ema_dx: float = 0.0
        self._ema_dy: float = 0.0

        # Tek karedeki maksimum kabul edilebilir yer değiştirme (metre)
        self._max_displacement_per_frame: float = 5.0

        # GPS → Optik Akış geçiş takibi
        self._was_gps_healthy: bool = False

        # Önceki kare bilgisi (Optik Akış için)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._initial_point_count: int = 0  # İlk tespit edilen köşe sayısı

        # ----- Shi-Tomasi Köşe Tespit Parametreleri -----
        self._feature_params = dict(
            maxCorners=100,       # Takip edilecek maksimum köşe sayısı (az ama yeterli)
            qualityLevel=0.01,    # Minimum köşe kalitesi (0-1)
            minDistance=40,       # Köşeler arası minimum mesafe (dengeli dağılım)
            blockSize=7,          # Analiz penceresi boyutu
        )

        # ----- Lucas-Kanade Optik Akış Parametreleri -----
        self._lk_params = dict(
            winSize=(21, 21),     # Arama penceresi boyutu
            maxLevel=3,           # Piramit seviyesi (görüntü ölçekleme)
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,    # Maksimum iterasyon
                0.01,  # Minimum hata eşiği
            ),
        )

        self.log.info("Visual Odometry başlatıldı — Başlangıç: (0, 0, 0)")

    # =========================================================================
    #  ANA GÜNCELLEME FONKSİYONU
    # =========================================================================

    def update(
        self,
        frame_ctx: "FrameContext",
        server_data: Dict,
    ) -> Dict[str, float]:
        """
        Bir video karesi ve sunucu verisi ile pozisyonu günceller.

        Hibrit Mantık:
            IF gps_health == 1:
                → Sunucu verisini kullan, gri kareyi son GPS karesi olarak sakla
            IF gps_health == 0:
                → Optik Akış ile piksel kaydırma hesapla, metreye çevir

        Optimizasyon: GPS modunda Shi-Tomasi köşe tespiti YAPILMAZ.
        Referans kare sadece GPS→OF geçişinde bir kez oluşturulur.

        Args:
            frame_ctx: FrameContext objesi (paylaşımlı işlemler için).
            server_data: Sunucudan gelen kare verisi.

        Returns:
            Güncel pozisyon: {"x": float, "y": float, "z": float}
        """
        gps_health = server_data.get("gps_health", 0)

        # Merkezi FrameContext objesinden gri kareyi al (tekrar tekrar çevrilmesini önler)
        if isinstance(frame_ctx, np.ndarray):
            gray = cv2.cvtColor(frame_ctx, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_ctx.gray

        if gps_health == 1:
            # ========== GPS SAĞLIKLI: Sunucu verisini kullan ==========
            self._update_from_gps(server_data)
            # Gri kareyi sakla ama köşe tespiti YAPMA (performans)
            # GPS→OF geçişinde bu kare referans olarak kullanılacak
            self._prev_gray = gray
            self._was_gps_healthy = True

        else:
            # ========== GPS SAĞLIKSIZ: Optik Akış ile kestirim ==========
            if self._was_gps_healthy:
                # GPS → OF geçişi: referans kareyi şimdi oluştur
                self._update_reference_frame(
                    self._prev_gray if self._prev_gray is not None else gray
                )
                self._was_gps_healthy = False
                self.log.info("GPS → Optik Akış geçişi — referans kare oluşturuldu")

            if self._prev_gray is not None and self._prev_points is not None:
                self._update_from_optical_flow(gray, server_data)
            else:
                self.log.warn(
                    "GPS yok ve referans kare henüz oluşmadı — "
                    "pozisyon güncellenemiyor"
                )
                self._update_reference_frame(gray)

        return self.get_position()

    # =========================================================================
    #  GPS TABANLI GÜNCELLEME
    # =========================================================================

    def _update_from_gps(self, server_data: Dict) -> None:
        """
        GPS sağlıklı olduğunda sunucu verisinden pozisyonu günceller
        ve sistemi kalibre eder.

        Args:
            server_data: Sunucudan gelen veri (translation_x/y/z içermeli).
        """
        new_x = float(server_data.get("translation_x", self.position["x"]))
        new_y = float(server_data.get("translation_y", self.position["y"]))
        new_z = float(server_data.get("translation_z", self.position["z"]))

        self.position["x"] = new_x
        self.position["y"] = new_y
        self.position["z"] = new_z

        # GPS irtifasını kaydet (optik akış fallback için)
        if new_z > 0:
            self._last_gps_altitude = new_z

        # Kalibrasyon referansını kaydet
        self._last_gps_position = {
            "x": new_x,
            "y": new_y,
            "z": new_z,
        }

        # EMA durumunu sıfırla (GPS→OF geçişinde temiz başlangıç)
        self._ema_dx = 0.0
        self._ema_dy = 0.0

        self.log.debug(
            f"GPS güncelleme → X:{new_x:.2f}m Y:{new_y:.2f}m Z:{new_z:.2f}m"
        )

    # =========================================================================
    #  OPTİK AKIŞ TABANLI GÜNCELLEME
    # =========================================================================

    def _update_from_optical_flow(
        self,
        gray: np.ndarray,
        server_data: Dict,
    ) -> None:
        """
        GPS kesildiğinde Lucas-Kanade Sparse Optical Flow ile
        piksel kaydırma hesaplar ve bunu metreye çevirir.

        İyileştirme: Referans kare sadece takip edilen nokta sayısı
        başlangıcın %50'sinin altına düştüğünde yenilenir (her karede değil).

        Args:
            gray: Güncel karenin gri tonlamalı hali.
            server_data: Sunucu verisi (irtifa bilgisi için).
        """
        if self._prev_points is None or len(self._prev_points) < 10:
            # Yeterli köşe yoksa yeniden tespit et
            self._update_reference_frame(gray)
            self.log.warn("Yetersiz köşe noktası — yeniden tespit ediliyor")
            return

        if self._prev_points is None:
            self._update_reference_frame(gray)
            return

        # ------ 1) Lucas-Kanade ile noktaları takip et ------
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        if next_points is None or status is None:
            self._update_reference_frame(gray)
            self.log.warn("Optik Akış başarısız — referans kare yenileniyor")
            return

        # ------ 2) Başarılı takip edilen noktaları filtrele ------
        mask = status.flatten() == 1
        
        if self._prev_points is None:
            return

        good_old = self._prev_points[mask].reshape(-1, 2)
        good_new = next_points[mask].reshape(-1, 2)

        if len(good_new) < 5:
            self._update_reference_frame(gray)
            self.log.warn("Başarılı takip sayısı az — referans yenileniyor")
            return

        # ------ 3) Ortalama piksel kaydırmasını hesapla ------
        dx_pixels = float(np.median(good_new[:, 0] - good_old[:, 0]))
        dy_pixels = float(np.median(good_new[:, 1] - good_old[:, 1]))

        # ------ 4) Piksel → Metre dönüşümü ------
        # Son bilinen GPS irtifasını kullan (sabit 50m yerine)
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

        # ------ 5) EMA yumuşatma ------
        alpha = self._ema_alpha
        self._ema_dx = alpha * dx_meters + (1 - alpha) * self._ema_dx
        self._ema_dy = alpha * dy_meters + (1 - alpha) * self._ema_dy

        # ------ 6) Aykırı değer sınırlama ------
        cap = self._max_displacement_per_frame
        smooth_dx = max(-cap, min(cap, self._ema_dx))
        smooth_dy = max(-cap, min(cap, self._ema_dy))

        # ------ 7) Pozisyonu güncelle ------
        self.position["x"] += smooth_dx
        self.position["y"] += smooth_dy

        self.log.debug(
            f"Optik Akış → dX:{dx_meters:.3f}m dY:{dy_meters:.3f}m | "
            f"Piksel: ({dx_pixels:.1f}, {dy_pixels:.1f}) | "
            f"İrtifa: {altitude:.1f}m | "
            f"Takip: {len(good_new)}/{len(self._prev_points)} nokta"
        )

        # ------ 6) Referans karesini akıllıca güncelle ------
        # Takip edilen nokta sayısı başlangıcın %50'sinin altına düştüyse yenile
        if (
            self._initial_point_count > 0
            and len(good_new) < self._initial_point_count * 0.5
        ):
            self._update_reference_frame(gray)
            self.log.debug("Köşe noktası kaybı %50 üzeri — referans yenilendi")
        else:
            # Mevcut noktaları sonraki kare için güncelle (yeniden tespit etme)
            self._prev_gray = gray.copy()
            self._prev_points = good_new.reshape(-1, 1, 2)

    # =========================================================================
    #  PİKSEL → METRE DÖNÜŞÜMÜ
    # =========================================================================

    def _pixel_to_meter(
        self,
        dx_px: float,
        dy_px: float,
        altitude: float,
    ) -> Tuple[float, float]:
        """
        Piksel cinsinden kaydırmayı metreye çevirir.

        Formül (pinhole kamera modeli):
            dx_m = dx_px * altitude / focal_length
            dy_m = dy_px * altitude / focal_length

        Args:
            dx_px: X eksenindeki piksel kaydırma.
            dy_px: Y eksenindeki piksel kaydırma.
            altitude: Yer düzlemine olan irtifa (metre).

        Returns:
            (dx_meter, dy_meter) tuple.
        """
        focal = Settings.FOCAL_LENGTH_PX
        if focal <= 0:
            focal = 800.0  # Güvenli varsayılan

        dx_m = dx_px * altitude / focal
        dy_m = dy_px * altitude / focal

        return dx_m, dy_m

    # =========================================================================
    #  REFERANS KARE GÜNCELLEMESİ
    # =========================================================================

    def _update_reference_frame(self, gray: np.ndarray) -> None:
        """
        Optik Akış için referans kareyi ve köşe noktalarını günceller.

        Shi-Tomasi Corner Detection ile yeni köşeler tespit eder.

        Args:
            gray: Gri tonlamalı görüntü.
        """
        self._prev_gray = gray.copy()
        self._prev_points = cv2.goodFeaturesToTrack(
            gray, **self._feature_params
        )

        n_points = len(self._prev_points) if self._prev_points is not None else 0
        self._initial_point_count = n_points
        self.log.debug(f"Referans kare güncellendi — {n_points} köşe noktası")

    # =========================================================================
    #  POZİSYON SORGULAMA
    # =========================================================================

    def get_position(self) -> Dict[str, float]:
        """
        Güncel pozisyonu döndürür.

        Returns:
            {"x": float, "y": float, "z": float} — metre cinsinden.
        """
        return {
            "x": round(self.position["x"], 4),
            "y": round(self.position["y"], 4),
            "z": round(self.position["z"], 4),
        }

    def reset(self) -> None:
        """
        Pozisyonu ve referans kareyi sıfırlar.
        Yeni oturum başlatmada kullanılır.
        """
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._prev_gray = None
        self._prev_points = None
        self._last_gps_position = None
        self._initial_point_count = 0
        self.log.info("Visual Odometry sıfırlandı → (0, 0, 0)")
