"""
TEKNOFEST Havacılıkta Yapay Zeka - Nesne Tespit Modülü (Görev 1)
=================================================================
YOLOv8 tabanlı nesne tespiti ve İniş Uygunluğu (Landing Status) mantığı.

İniş Uygunluğu Algoritması (teknofest_context.md'den):
    1. Taşıt (0) ve İnsan (1) → landing_status = "-1" (iniş alanı değil)
    2. UAP (2) ve UAİ (3) için:
       a. Bounding box kadrajın kenarına değiyorsa → "0" (alan tam görünmüyor)
       b. Alan tamamen kadrajda VE üzerinde İnsan/Taşıt varsa (kesişim kontrolü) → "0"
       c. Alan tamamen kadrajda VE üzerinde engel yoksa → "1" (uygun)

Kullanım:
    from src.detection import ObjectDetector
    detector = ObjectDetector()
    detections = detector.detect(frame)
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import Settings
from src.utils import Logger


class ObjectDetector:
    """
    YOLOv8 tabanlı nesne tespit sınıfı.

    CUDA üzerinde çalışır, COCO sınıflarını TEKNOFEST sınıflarına dönüştürür
    ve UAP/UAİ alanları için İniş Uygunluğu hesabı yapar.

    Attributes:
        model: YOLOv8 model nesnesi.
        device: Kullanılan cihaz ('cuda' veya 'cpu').
        log: Logger nesnesi.
    """

    def __init__(self) -> None:
        """
        YOLOv8 modelini yükler ve CUDA cihazına taşır.

        Model dosyası Settings.MODEL_PATH'ten yerel diskten yüklenir.
        CUDA kullanılamıyorsa CPU'ya düşer ve uyarı verir.
        İlk kare gecikmesini önlemek için warmup inference yapılır.
        """
        self.log = Logger("Detector")
        self._frame_count: int = 0
        self._use_half: bool = False

        # Cihaz Seçimi
        if Settings.DEVICE == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.log.success(f"GPU aktif: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.device = "cpu"
            self.log.warn("CUDA bulunamadı! CPU modunda çalışılıyor (yavaş olacak)")

        # Model Yükleme (Yerel Diskten - OFFLINE MODE)
        self.log.info(f"YOLOv8 modeli yükleniyor: {Settings.MODEL_PATH}")
        try:
            if not os.path.exists(Settings.MODEL_PATH):
                raise FileNotFoundError(
                    f"Model dosyası bulunamadı: {Settings.MODEL_PATH}\n"
                    f"  → 'models/' dizinine yolov8n.pt dosyasını kopyalayın."
                )
            self.model = YOLO(Settings.MODEL_PATH)
            self.model.to(self.device)

            # FP16 Yarı Hassasiyet — GPU'da ~%40 hız artışı
            if self.device == "cuda" and Settings.HALF_PRECISION:
                self._use_half = True
                self.log.info("FP16 (Half Precision) aktif — hız optimizasyonu ✓")

            self.log.success("Model başarıyla yüklendi ✓")
        except Exception as e:
            self.log.error(f"Model yükleme hatası: {e}")
            raise RuntimeError(f"YOLOv8 modeli yüklenemedi: {e}")

        # Warmup — ilk kare gecikmesini önle
        self._warmup()

        # CLAHE nesnesi (ön-işleme için)
        if Settings.CLAHE_ENABLED:
            self._clahe = cv2.createCLAHE(
                clipLimit=Settings.CLAHE_CLIP_LIMIT,
                tileGridSize=(
                    Settings.CLAHE_TILE_SIZE,
                    Settings.CLAHE_TILE_SIZE,
                ),
            )
            self.log.info("CLAHE kontrast iyileştirme aktif ✓")
        else:
            self._clahe = None

    def _warmup(self) -> None:
        """
        Model ısınması — GPU belleğini hazırlar ve ilk kare gecikmesini önler.

        Dummy (boş) bir tensor ile birkaç inference yaparak CUDA kernel'larını
        ve cuDNN autotuner'ı önceden başlatır.
        """
        self.log.info(f"Model ısınması başlıyor ({Settings.WARMUP_ITERATIONS} iterasyon)...")
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                for i in range(Settings.WARMUP_ITERATIONS):
                    self.model.predict(
                        source=dummy,
                        imgsz=Settings.INFERENCE_SIZE,
                        conf=Settings.CONFIDENCE_THRESHOLD,
                        device=self.device,
                        verbose=False,
                        save=False,
                        half=self._use_half,
                    )
            self.log.success(f"Model ısınması tamamlandı ✓")
        except Exception as e:
            self.log.warn(f"Warmup sırasında hata (görmezden geliniyor): {e}")

    # =========================================================================
    #  ANA TESPİT FONKSİYONU
    # =========================================================================

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Bir video karesinde nesne tespiti yapar ve sonuçları TEKNOFEST formatında döndürür.

        SAHI aktifse:
            1. Full-frame inference (büyük nesneler)
            2. Sliced inference (küçük nesneler — tepeden görünüm)
            3. NMS ile birleştirme

        Args:
            frame: BGR formatlı OpenCV görüntüsü (numpy array).

        Returns:
            Tespit edilen nesnelerin listesi (TEKNOFEST formatında dict'ler).
        """
        try:
            # ---- 0) Ön-İşleme (CLAHE + Sharpening) ----
            processed = self._preprocess(frame)

            # ---- 1) Inference (SAHI veya Standard) ----
            if Settings.SAHI_ENABLED:
                raw_detections = self._sahi_detect(processed)
            else:
                raw_detections = self._standard_inference(processed)

            # ---- 2) Post-Processing Filtreleri ----
            raw_detections = self._post_filter(raw_detections)

            # ---- 3) İniş Uygunluğu Hesaplaması ----
            frame_h, frame_w = frame.shape[:2]
            final_detections = self._determine_landing_status(
                raw_detections, frame_w, frame_h
            )

            # ---- 4) Dahili alanları temizle, TEKNOFEST formatı döndür ----
            output: List[Dict] = []
            for det in final_detections:
                output.append({
                    "cls": det["cls"],
                    "landing_status": det["landing_status"],
                    "top_left_x": det["top_left_x"],
                    "top_left_y": det["top_left_y"],
                    "bottom_right_x": det["bottom_right_x"],
                    "bottom_right_y": det["bottom_right_y"],
                    "confidence": det["confidence"],
                })

            # Debug log
            if Settings.DEBUG:
                cls_counts = Counter(d["cls"] for d in output)
                self.log.debug(
                    f"Tespit: {len(output)} nesne "
                    f"(Taşıt: {cls_counts.get('0', 0)}, "
                    f"İnsan: {cls_counts.get('1', 0)}, "
                    f"UAP: {cls_counts.get('2', 0)}, "
                    f"UAİ: {cls_counts.get('3', 0)})"
                )

            # ---- GPU Bellek Temizliği (periyodik) ----
            self._frame_count += 1
            if (
                self.device == "cuda"
                and self._frame_count % Settings.GPU_CLEANUP_INTERVAL == 0
            ):
                torch.cuda.empty_cache()
                self.log.debug("GPU bellek temizlendi (empty_cache)")

            return output

        except Exception as e:
            self.log.error(f"Tespit hatası: {e}")
            return []

    # =========================================================================
    #  STANDARD INFERENCE (Tek Geçiş)
    # =========================================================================

    def _standard_inference(self, frame: np.ndarray) -> List[Dict]:
        """Standart tam-kare inference — büyük nesneler için."""
        with torch.no_grad():
            results = self.model.predict(
                source=frame,
                imgsz=Settings.INFERENCE_SIZE,
                conf=Settings.CONFIDENCE_THRESHOLD,
                iou=Settings.NMS_IOU_THRESHOLD,
                device=self.device,
                verbose=False,
                save=False,
                half=self._use_half,
                agnostic_nms=Settings.AGNOSTIC_NMS,
                max_det=Settings.MAX_DETECTIONS,
                augment=Settings.AUGMENTED_INFERENCE,
            )
        return self._parse_results(results)

    # =========================================================================
    #  SAHI — Slicing Aided Hyper Inference
    # =========================================================================

    def _sahi_detect(self, frame: np.ndarray) -> List[Dict]:
        """
        SAHI: Tam-kare + parçalı inference → NMS ile birleştirme.

        Neden gerekli:
            Drone 50m'den çekerken araçlar ~30px, insanlar ~15px.
            1280px inference'ta bile çok küçükler.
            640×640 parçalara bölünce, aynı araç ~120px olur → tespit edilir.

        Strateji:
            1) Full-frame @ INFERENCE_SIZE → büyük nesneleri yakalar
            2) Sliced @ SAHI_SLICE_SIZE → küçük nesneleri yakalar
            3) Tüm sonuçları NMS ile birleştir → duplikasyonu önle
        """
        all_detections: List[Dict] = []

        # ---- 1) Full-frame inference (büyük nesneler) ----
        full_dets = self._standard_inference(frame)
        all_detections.extend(full_dets)

        # ---- 2) Sliced inference (küçük nesneler) ----
        slice_dets = self._sliced_inference(frame)
        all_detections.extend(slice_dets)

        # ---- 3) NMS ile birleştirme (duplikasyonu önle) ----
        if len(all_detections) > 0:
            all_detections = self._merge_detections_nms(all_detections)

        return all_detections

    def _sliced_inference(self, frame: np.ndarray) -> List[Dict]:
        """
        Görüntüyü örtüşen parçalara böler ve her parçada inference yapar.

        Koordinatlar orijinal görüntü koordinatlarına geri dönüştürülür.
        """
        h, w = frame.shape[:2]
        slice_size = Settings.SAHI_SLICE_SIZE
        overlap = Settings.SAHI_OVERLAP_RATIO
        step = int(slice_size * (1 - overlap))

        all_slice_dets: List[Dict] = []

        with torch.no_grad():
            for y_start in range(0, h, step):
                for x_start in range(0, w, step):
                    # Parça sınırlarını hesapla
                    x_end = min(x_start + slice_size, w)
                    y_end = min(y_start + slice_size, h)

                    # Çok küçük kenar parçalarını atla
                    if (x_end - x_start) < slice_size // 2 or \
                       (y_end - y_start) < slice_size // 2:
                        continue

                    # Parçayı kes
                    tile = frame[y_start:y_end, x_start:x_end]

                    # Inference (parça boyutunda, tile'ın kendi boyutu)
                    results = self.model.predict(
                        source=tile,
                        imgsz=slice_size,
                        conf=Settings.CONFIDENCE_THRESHOLD,
                        iou=Settings.NMS_IOU_THRESHOLD,
                        device=self.device,
                        verbose=False,
                        save=False,
                        half=self._use_half,
                        agnostic_nms=Settings.AGNOSTIC_NMS,
                        max_det=Settings.MAX_DETECTIONS,
                    )

                    # Koordinatları orijinal frame'e dönüştür
                    tile_dets = self._parse_results(results)
                    for det in tile_dets:
                        det["top_left_x"] += x_start
                        det["top_left_y"] += y_start
                        det["bottom_right_x"] += x_start
                        det["bottom_right_y"] += y_start
                        det["bbox"] = (
                            det["top_left_x"],
                            det["top_left_y"],
                            det["bottom_right_x"],
                            det["bottom_right_y"],
                        )
                    all_slice_dets.extend(tile_dets)

        return all_slice_dets

    # =========================================================================
    #  YARDIMCI METODLAR
    # =========================================================================

    def _parse_results(self, results) -> List[Dict]:
        """YOLO sonuçlarını COCO→TEKNOFEST dönüşümü ile parse eder."""
        detections: List[Dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coco_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = (
                    float(coords[0]), float(coords[1]),
                    float(coords[2]), float(coords[3]),
                )

                tf_id = Settings.COCO_TO_TEKNOFEST.get(coco_id, -1)
                if tf_id == -1:
                    continue

                detections.append({
                    "cls_int": tf_id,
                    "cls": str(tf_id),
                    "confidence": int(conf * 10000) / 10000,
                    "top_left_x": int(x1),
                    "top_left_y": int(y1),
                    "bottom_right_x": int(x2),
                    "bottom_right_y": int(y2),
                    "bbox": (x1, y1, x2, y2),
                })
        return detections

    @staticmethod
    def _merge_detections_nms(detections: List[Dict]) -> List[Dict]:
        """
        Birden fazla kaynaktan gelen tespitleri NMS ile birleştirir.

        Full-frame ve sliced sonuçlarda aynı nesne birden fazla kez
        tespit edilebilir. Bu metod IoU tabanlı NMS ile duplikasyonları kaldırır.
        En yüksek confidence'lı tespit korunur.
        """
        if not detections:
            return detections

        # NumPy array'lere dönüştür (NMS için)
        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

        # Sınıf bazlı NMS (farklı sınıflar birbirini bastırmasın)
        class_ids = np.array([d["cls_int"] for d in detections], dtype=np.int32)
        unique_classes = np.unique(class_ids)

        keep_indices: List[int] = []
        for cls_id in unique_classes:
            cls_mask = class_ids == cls_id
            cls_indices = np.where(cls_mask)[0]

            if len(cls_indices) == 0:
                continue

            cls_boxes = boxes[cls_indices]
            cls_scores = scores[cls_indices]

            # Greedy NMS
            nms_keep = ObjectDetector._nms_greedy(
                cls_boxes, cls_scores, Settings.SAHI_MERGE_IOU
            )
            # NMS sonrası indeksleri güncelle
            nms_indices = cls_indices[nms_keep]
            keep_indices.extend(nms_indices.tolist())

        nms_results = [detections[i] for i in keep_indices]

        # 2. Adım: Containment Suppression (İç içe geçen kutuları temizle)
        # Örn: Bacak (küçük kutu) Vücut (büyük kutu) içindeyse, küçüğü sil
        return ObjectDetector._suppress_contained(nms_results)

    @staticmethod
    def _suppress_contained(detections: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """
        Bir kutu diğerinin içindeyse (veya büyük oranda örtüşüyorsa) küçüğü siler.
        Standart NMS'in aksine, IoU yerine 'Intersection over Small Area' kullanır.
        """
        if not detections:
            return []

        # Alanlarına göre büyükten küçüğe sırala
        detections.sort(key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]), reverse=True)
        
        keep = []
        is_suppressed = [False] * len(detections)

        for i in range(len(detections)):
            if is_suppressed[i]:
                continue
            
            box_a = detections[i]["bbox"]
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            keep.append(detections[i])

            for j in range(i + 1, len(detections)):
                if is_suppressed[j]:
                    continue
                
                # Sadece aynı sınıfları kontrol et (İnsan içindeki İnsanı sil, Araç içindeki İnsanı silme)
                if detections[i]["cls_int"] != detections[j]["cls_int"]:
                    continue

                box_b = detections[j]["bbox"]
                
                # Kesişim alanı
                inter_x1 = max(box_a[0], box_b[0])
                inter_y1 = max(box_a[1], box_b[1])
                inter_x2 = min(box_a[2], box_b[2])
                inter_y2 = min(box_a[3], box_b[3])
                
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                
                if inter_area == 0:
                    continue

                # Küçük kutunun alanı (box_b çünkü sıralı gidiyoruz, j >= i, ama alan büyükten küçüğe)
                # DİKKAT: Sıralama Büyük -> Küçük. Yani i Büyük, j Küçük.
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                
                # IOS hesapla: Kesişim / Küçük Alan
                ios = inter_area / area_b if area_b > 0 else 0

                if ios > threshold:
                    is_suppressed[j] = True  # Küçüğü bastır

        return keep

    @staticmethod
    def _nms_greedy(
        boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> List[int]:
        """Greedy NMS implementasyonu (sınıf-agnostik)."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            # IoU hesapla
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / np.maximum(union, 1e-6)

            # IoU düşük olanları koru
            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    # =========================================================================
    #  ÖN-İŞLEME (PREPROCESSING)
    # =========================================================================

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Drone görüntülerini tespit öncesi iyileştirir.

        İşlemler:
            1. CLAHE: Adaptif kontrast iyileştirme (LAB renk uzayında L kanalına)
               → karanlık/gölgeli bölgelerdeki insanları ortaya çıkarır
            2. Sharpening: Hafif keskinleştirme (unsharp mask)
               → uzaktaki küçük nesnelerin kenarlarını belirginleştirir

        Args:
            frame: BGR formatlı OpenCV görüntüsü.

        Returns:
            İyileştirilmiş BGR görüntüsü.
        """
        result = frame

        # ---- CLAHE Kontrast İyileştirme ----
        if self._clahe is not None:
            # LAB renk uzayına çevir (L = parlaklık kanayı)
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Sadece L (parlaklık) kanalına CLAHE uygula
            l_enhanced = self._clahe.apply(l_channel)

            # Kanalları birleştir ve BGR'ye dön
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # ---- Hafif Sharpening (Unsharp Mask) ----
        # Gaussian bulanıklaştır → orijinalden çıkar → ekle
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=2.0)
        result = cv2.addWeighted(result, 1.3, blurred, -0.3, 0)

        return result

    # =========================================================================
    #  POST-PROCESSING FİLTRELERİ
    # =========================================================================

    @staticmethod
    def _post_filter(detections: List[Dict]) -> List[Dict]:
        """
        False positive'leri azaltmak için tespit sonrası filtreler.

        Filtreler:
            1. Minimum bbox boyutu: Çok küçük tespitler (< MIN_BBOX_SIZE px) → kaldır
            2. Aşırı aspect ratio: 6:1'den fazla uzun/geniş → kaldır (artefakt)

        Args:
            detections: Ham tespit listesi.

        Returns:
            Filtrelenmiş tespit listesi.
        """
        filtered: List[Dict] = []
        min_size = Settings.MIN_BBOX_SIZE

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1

            # Minimum boyut kontrolü
            if w < min_size or h < min_size:
                continue

            # Maksimum boyut kontrolü (Binaları/çatıları elemek için)
            # 50m irtifada 300px'den büyük bir şey araç olamaz (Otobüs bile ~200px)
            if w > Settings.MAX_BBOX_SIZE or h > Settings.MAX_BBOX_SIZE:
                continue

            # Aşırı aspect ratio kontrolü (> 6:1)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 6.0:
                continue

            filtered.append(det)

        return filtered

    # =========================================================================
    #  İNİŞ UYGUNLUĞU (LANDING STATUS) MANTIĞI
    # =========================================================================

    def _determine_landing_status(
        self,
        detections: List[Dict],
        frame_w: int,
        frame_h: int,
    ) -> List[Dict]:
        """
        Tespit edilen nesneler için iniş durumunu belirler.

        Mantık (teknofest_context.md - Bölüm 4.6):
            - Taşıt (0) ve İnsan (1): landing_status = "-1" (sabit)
            - UAP (2) ve UAİ (3):
                a) bbox kadrajın kenarına değiyorsa → "0" (alan tam görünmüyor)
                b) Üzerinde taşıt/insan varsa (kesişim kontrolü) → "0"
                c) Yukarıdaki koşullar sağlanmıyorsa → "1" (uygun)

        ÖNEMLİ: Engel kontrolünde IoU değil "intersection-over-area-of-landing-zone"
        kullanılır. Çünkü şartname "alanın **üzerinde** nesne var mı" soruyor —
        küçük bir insan büyük iniş alanının üzerinde olsa IoU çok düşük çıkar
        ama yine de uygun değildir.

        Args:
            detections: Ham tespit listesi (bbox alanı dahil).
            frame_w: Görüntü genişliği (piksel).
            frame_h: Görüntü yüksekliği (piksel).

        Returns:
            landing_status alanı doldurulmuş tespit listesi.
        """
        # Taşıt ve İnsan listesini ayır (engel olarak kullanılacak)
        obstacles: List[Tuple[float, float, float, float]] = []
        for det in detections:
            if det["cls_int"] in (Settings.CLASS_TASIT, Settings.CLASS_INSAN):
                obstacles.append(det["bbox"])

        # Her nesne için iniş durumunu belirle
        for det in detections:
            cls_id = det["cls_int"]

            if cls_id in (Settings.CLASS_TASIT, Settings.CLASS_INSAN):
                # ------ Taşıt veya İnsan: İniş alanı değil ------
                det["landing_status"] = Settings.LANDING_NOT_AREA

            elif cls_id in (Settings.CLASS_UAP, Settings.CLASS_UAI):
                # ------ UAP veya UAİ: İniş uygunluğu hesapla ------
                bbox = det["bbox"]

                # (a) Alan kadrajın kenarına değiyor mu?
                if self._is_touching_edge(bbox, frame_w, frame_h):
                    det["landing_status"] = Settings.LANDING_NOT_SUITABLE
                    self.log.debug(
                        f"  UAP/UAİ kenar temas → uygun değil"
                    )
                    continue

                # (b) Üzerinde engel var mı? (Kesişim kontrolü)
                has_obstacle = False
                for obs_bbox in obstacles:
                    overlap_ratio = self._intersection_over_area(bbox, obs_bbox)
                    if overlap_ratio > Settings.LANDING_IOU_THRESHOLD:
                        has_obstacle = True
                        self.log.debug(
                            f"  UAP/UAİ üzerinde engel (overlap={overlap_ratio:.3f}) → uygun değil"
                        )
                        break

                if has_obstacle:
                    det["landing_status"] = Settings.LANDING_NOT_SUITABLE
                else:
                    # (c) Alan tamamen kadrajda ve engelsiz → uygun
                    det["landing_status"] = Settings.LANDING_SUITABLE
                    self.log.debug("  UAP/UAİ → iniş uygun ✓")

            else:
                det["landing_status"] = Settings.LANDING_NOT_AREA

        return detections

    # =========================================================================
    #  KESİŞİM HESAPLAMA
    # =========================================================================

    @staticmethod
    def _intersection_over_area(
        landing_box: Tuple[float, float, float, float],
        obstacle_box: Tuple[float, float, float, float],
    ) -> float:
        """
        İniş alanı ile engel arasındaki kesişimin, iniş alanına oranını hesaplar.

        Bu, standart IoU'dan farklıdır: küçük bir engel (insan) büyük bir iniş
        alanının üzerinde olsa bile IoU çok düşük çıkar. Ancak bu hesaplama
        iniş alanının ne kadarının engelle örtüştüğünü doğru ölçer.

        Formül:
            overlap_ratio = intersection_area / landing_area

        Args:
            landing_box: İniş alanı (x1, y1, x2, y2).
            obstacle_box: Engel nesnesi (x1, y1, x2, y2).

        Returns:
            0.0 ile 1.0 arasında kesişim oranı.
        """
        # Kesişim alanının köşeleri
        inter_x1 = max(landing_box[0], obstacle_box[0])
        inter_y1 = max(landing_box[1], obstacle_box[1])
        inter_x2 = min(landing_box[2], obstacle_box[2])
        inter_y2 = min(landing_box[3], obstacle_box[3])

        # Kesişim alanı
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        # İniş alanının toplam alanı
        landing_area = (
            max(0.0, landing_box[2] - landing_box[0])
            * max(0.0, landing_box[3] - landing_box[1])
        )

        if landing_area == 0:
            return 0.0

        return inter_area / landing_area


    # =========================================================================
    #  KENAR TEMAS KONTROLÜ
    # =========================================================================

    @staticmethod
    def _is_touching_edge(
        bbox: Tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
        margin: int = 5,
    ) -> bool:
        """
        Bounding box'ın kadrajın kenarına değip değmediğini kontrol eder.

        Şartnameye göre: UAP/UAİ alanının tamamı kadraj içinde olmalıdır,
        aksi halde iniş durumu "uygun" olamaz.

        Args:
            bbox: (x1, y1, x2, y2) formatında bounding box.
            frame_w: Görüntü genişliği.
            frame_h: Görüntü yüksekliği.
            margin: Kenar toleransı (piksel).

        Returns:
            Kenarına değiyorsa True.
        """
        x1, y1, x2, y2 = bbox
        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= (frame_w - margin)
            or y2 >= (frame_h - margin)
        )
