"""
TEKNOFEST Havacılıkta Yapay Zeka - Görev 3: Referans Obje Tespiti (Image Matching)
====================================================================================
Oturum başında verilen referans obje görüntülerini video karelerinde tespit eder.

Yaklaşım:
    1. Referans obje görüntülerinden feature descriptor çıkar (ORB veya SIFT)
    2. Her video karesinde multi-scale sliding window + feature matching
    3. Eşik üstü benzerlik → bounding box + object_id olarak raporla

Kullanım:
    matcher = ImageMatcher()
    matcher.load_references(reference_images)   # Oturum başında
    results = matcher.match(frame)              # Her karede

    # results formatı:
    # [{"object_id": 1, "top_left_x": ..., "top_left_y": ...,
    #   "bottom_right_x": ..., "bottom_right_y": ..., "similarity": 0.85}, ...]
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings
from src.utils import Logger


class ReferenceObject:
    """Tek bir referans objenin feature bilgilerini tutar."""

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
    """
    Referans obje eşleştirme motoru.

    Oturum başında verilen referans obje fotoğraflarını ORB/SIFT feature
    descriptor'ları ile indeksler. Her video karesinde bu descriptor'ları
    arayarak eşleşen nesnelerin bounding box koordinatlarını döndürür.

    Şartname Gereksinimleri:
        - Farklı kameradan (termal→RGB) çekilmiş olabilir
        - Farklı açı/irtifadan çekilmiş olabilir
        - Uydu görüntüsünden alınmış olabilir
        - Yer yüzeyinden çekilmiş olabilir
        - Çeşitli görüntü işleme uygulanmış olabilir
    """

    def __init__(self) -> None:
        self.log = Logger("Task3")
        self.references: List[ReferenceObject] = []
        self._frame_counter: int = 0

        # Feature detector seçimi
        method = Settings.TASK3_FEATURE_METHOD.upper()
        if method == "SIFT":
            self.detector = cv2.SIFT_create()
            self.norm_type = cv2.NORM_L2
            self.log.info("Feature method: SIFT (daha robust, yavaş)")
        else:
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.norm_type = cv2.NORM_HAMMING
            self.log.info("Feature method: ORB (hızlı, offline-uyumlu)")

        # BFMatcher — cross-check ile daha güvenilir eşleşme
        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)

        self.log.info(
            f"ImageMatcher initialized | "
            f"similarity_threshold={Settings.TASK3_SIMILARITY_THRESHOLD} | "
            f"fallback_threshold={Settings.TASK3_FALLBACK_THRESHOLD}"
        )

    def load_references(self, reference_images: List[Dict[str, Any]]) -> int:
        """
        Referans obje görüntülerini yükler ve feature'larını çıkarır.

        Args:
            reference_images: Her biri {"object_id": int, "image": np.ndarray}
                              veya {"object_id": int, "path": str} olan dict listesi.

        Returns:
            Başarıyla yüklenen referans sayısı.
        """
        self.references.clear()
        loaded = 0

        for ref_data in reference_images:
            object_id = ref_data.get("object_id", loaded + 1)
            label = ref_data.get("label", f"ref_{object_id}")

            # Görüntüyü yükle
            if "image" in ref_data and ref_data["image"] is not None:
                image = ref_data["image"]
            elif "path" in ref_data and os.path.isfile(ref_data["path"]):
                image = cv2.imread(ref_data["path"])
                if image is None:
                    self.log.warn(f"Referans obje okunamadı: {ref_data['path']}")
                    continue
            else:
                self.log.warn(f"Referans obje #{object_id} için geçerli görüntü bulunamadı")
                continue

            # Gri tonlama + feature extraction
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 4:
                self.log.warn(
                    f"Referans obje #{object_id}: yetersiz feature ({len(keypoints) if keypoints else 0})"
                )
                continue

            ref_obj = ReferenceObject(
                object_id=object_id,
                image=image,
                keypoints=keypoints,
                descriptors=descriptors,
                label=label,
            )
            self.references.append(ref_obj)
            loaded += 1

            self.log.info(
                f"Referans #{object_id} yüklendi: "
                f"{ref_obj.w}x{ref_obj.h}px, {len(keypoints)} feature"
            )

        self.log.success(f"Toplam {loaded}/{len(reference_images)} referans obje yüklendi")
        return loaded

    def load_references_from_directory(self, directory: Optional[str] = None) -> int:
        """
        Dizindeki tüm görüntüleri referans obje olarak yükler.

        Args:
            directory: Referans obje dizini. None ise Settings.TASK3_REFERENCE_DIR kullanılır.

        Returns:
            Yüklenen referans sayısı.
        """
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
        """
        Bir video karesinde referans objeleri arar.

        Args:
            frame: BGR formatlı OpenCV görüntüsü.

        Returns:
            Tespit edilen referans objelerin listesi:
            [{"object_id": int, "top_left_x": float, "top_left_y": float,
              "bottom_right_x": float, "bottom_right_y": float}, ...]
        """
        self._frame_counter += 1

        if not self.references:
            return []

        # Frame feature extraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_kp, frame_desc = self.detector.detectAndCompute(gray, None)

        if frame_desc is None or len(frame_kp) < 4:
            return []

        results: List[Dict[str, Any]] = []

        for ref in self.references:
            bbox = self._match_reference(ref, frame_kp, frame_desc, gray.shape)
            if bbox is not None:
                results.append({
                    "object_id": ref.object_id,
                    "top_left_x": bbox[0],
                    "top_left_y": bbox[1],
                    "bottom_right_x": bbox[2],
                    "bottom_right_y": bbox[3],
                })

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
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Tek bir referans objeyi frame feature'ları ile eşleştirir.

        Lowe's ratio test ile güvenilir eşleşmeleri seçer,
        ardından homography ile bounding box hesaplar.

        Returns:
            (x1, y1, x2, y2) bounding box veya None.
        """
        if ref.descriptors is None:
            return None

        try:
            # KNN matching
            matches = self.matcher.knnMatch(ref.descriptors, frame_desc, k=2)
        except cv2.error:
            return None

        # Lowe's ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) < 2:
                continue
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Minimum eşleşme sayısı
        min_matches = max(4, int(len(ref.keypoints) * 0.05))
        if len(good_matches) < min_matches:
            return None

        # Benzerlik skoru hesapla
        similarity = len(good_matches) / max(1, len(ref.keypoints))

        # Eşik kontrolü
        threshold = Settings.TASK3_SIMILARITY_THRESHOLD
        if self._frame_counter % Settings.TASK3_FALLBACK_INTERVAL == 0:
            threshold = Settings.TASK3_FALLBACK_THRESHOLD

        if similarity < threshold:
            return None

        # Homography ile bounding box hesapla
        try:
            src_pts = np.float32(
                [ref.keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [frame_kp[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            # Koliner noktalar homografi hesaplamasını bozar
            if len(np.unique(src_pts.reshape(-1, 2), axis=0)) < 4:
                return None
            if len(np.unique(dst_pts.reshape(-1, 2), axis=0)) < 4:
                return None

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None or M.shape != (3, 3):
                # Homografi başarısız: eşleşen noktalardan bounding rect
                self.log.warn("Homografi dejenere oldu, nokta bazlı bounding rect (fallback) çıkarıldı")
                pts = dst_pts.reshape(-1, 2)
            else:
                # Referans objenin köşelerini dönüştür
                h, w = ref.h, ref.w
                corners = np.float32(
                    [[0, 0], [w, 0], [w, h], [0, h]]
                ).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, M)
                pts = transformed.reshape(-1, 2)

            # Bounding box
            x1 = float(max(0, pts[:, 0].min()))
            y1 = float(max(0, pts[:, 1].min()))
            x2 = float(min(frame_shape[1] if len(frame_shape) > 1 else frame_shape[0], pts[:, 0].max()))
            y2 = float(min(frame_shape[0], pts[:, 1].max()))

            # Geçersiz bbox kontrolü
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < 5 or bbox_h < 5:
                return None
            if bbox_w > frame_shape[1] * 0.8 or bbox_h > frame_shape[0] * 0.8:
                return None

            return (x1, y1, x2, y2)

        except (cv2.error, ValueError, IndexError):
            return None

    @property
    def reference_count(self) -> int:
        """Yüklenmiş referans obje sayısı."""
        return len(self.references)

    @property
    def is_ready(self) -> bool:
        """Eşleştirme için en az bir referans yüklenmiş mi?"""
        return len(self.references) > 0

    def reset(self) -> None:
        """Tüm referansları ve frame sayacını sıfırlar."""
        self.references.clear()
        self._frame_counter = 0
        self.log.info("ImageMatcher reset")
