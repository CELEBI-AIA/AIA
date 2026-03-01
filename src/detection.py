"""YOLOv8 nesne tespiti + iniş uygunluğu (UAP/UAİ).
Model COCO/VisDrone vb. eğitilmiş olabilir; sınıflar TEKNOFEST (0,1,2,3) formatına map edilir."""

import os
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import Settings
from src.utils import Logger


class ObjectDetector:
    """YOLOv8 tespit, TEKNOFEST sınıf eşlemesi ve iniş uygunluğu."""

    def __init__(self) -> None:
        self.log = Logger("Detector")
        self._frame_count: int = 0
        self._use_half: bool = False
        self._class_map_mode: str = "unknown"
        self._model_class_map: Dict[int, int] = {}

        if Settings.DEVICE == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.log.success(f"GPU aktif: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.device = "cpu"
            self.log.warn("CUDA bulunamadı! CPU modunda çalışılıyor (yavaş olacak)")

        self.log.info(f"YOLOv8 modeli yükleniyor: {Settings.MODEL_PATH}")
        try:
            if not os.path.exists(Settings.MODEL_PATH):
                raise FileNotFoundError(
                    f"Model dosyası bulunamadı: {Settings.MODEL_PATH}\n"
                    f"  → 'models/' dizinine {os.path.basename(Settings.MODEL_PATH)} dosyasını kopyalayın."
                )
            self.model = YOLO(Settings.MODEL_PATH)
            self.model.to(self.device)
            self._configure_class_mapping()

            if self.device == "cuda" and Settings.HALF_PRECISION:
                self._use_half = True
                self.log.info("FP16 (Half Precision) aktif — hız optimizasyonu ✓")

            self.log.success("Model başarıyla yüklendi ✓")
        except Exception as e:
            self.log.error(f"Model yükleme hatası: {e}")
            raise RuntimeError(f"YOLOv8 modeli yüklenemedi: {e}")

        self._warmup()

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
            self.log.success("Model ısınması tamamlandı ✓")
        except Exception as e:
            self.log.warn(f"Warmup sırasında hata (görmezden geliniyor): {e}")

    @staticmethod
    def _normalize_label(label: str) -> str:
        text = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
        text = text.lower().strip().replace("-", " ").replace("_", " ")
        return " ".join(text.split())

    @staticmethod
    def _map_label_to_teknofest_id(label: str) -> int:
        normalized = ObjectDetector._normalize_label(label)
        if normalized in {"uap", "uap alan", "ucan araba park", "flying car park"}:
            return Settings.CLASS_UAP
        if normalized in {"uai", "uai alan", "uai alani", "ucan ambulans inis"}:
            return Settings.CLASS_UAI
        if normalized in {
            "insan", "human", "person", "pedestrian", "people", "man", "woman",
        }:
            return Settings.CLASS_INSAN
        if normalized in {
            "tasit", "vehicle", "car", "van", "truck", "bus", "train", "boat",
            "bicycle", "motorcycle", "motor", "tricycle", "awning tricycle",
        }:
            return Settings.CLASS_TASIT
        return -1

    def _configure_class_mapping(self) -> None:
        """Model sınıf isimlerine göre COCO/VisDrone/resmi eşleme seçer."""
        names_raw = getattr(self.model, "names", {})
        items: List[Tuple[int, str]] = []
        if isinstance(names_raw, dict):
            items = [(int(k), str(v)) for k, v in names_raw.items()]
        elif isinstance(names_raw, list):
            items = [(i, str(v)) for i, v in enumerate(names_raw)]

        name_based_map: Dict[int, int] = {}
        normalized_names: Dict[int, str] = {}
        for idx, label in items:
            normalized = self._normalize_label(label)
            normalized_names[idx] = normalized
            mapped = self._map_label_to_teknofest_id(label)
            if mapped != -1:
                name_based_map[idx] = mapped

        direct_ids = [0, 1, 2, 3]
        direct_ok = all(name_based_map.get(i, -1) == i for i in direct_ids)
        looks_like_coco = (
            normalized_names.get(0) == "person"
            and normalized_names.get(1) == "bicycle"
            and normalized_names.get(2) == "car"
        )

        if direct_ok:
            self._class_map_mode = "official_direct"
            self._model_class_map = {i: i for i in direct_ids}
        elif looks_like_coco:
            self._class_map_mode = "coco_remap"
            self._model_class_map = dict(Settings.COCO_TO_TEKNOFEST)
        else:
            self._class_map_mode = "name_based"
            self._model_class_map = name_based_map

        self.log.info(
            f"Sınıf eşleme modu: {self._class_map_mode} "
            f"(tanınan model sınıfı: {len(self._model_class_map)})"
        )

    def _map_model_class_to_teknofest(self, model_cls_id: int) -> int:
        return self._model_class_map.get(model_cls_id, -1)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        try:
            processed = self._preprocess(frame)
            if Settings.SAHI_ENABLED:
                raw_detections = self._sahi_detect(processed)
            else:
                raw_detections = self._standard_inference(processed)
            raw_detections = self._post_filter(raw_detections)
            frame_h, frame_w = frame.shape[:2]
            final_detections = self._determine_landing_status(
                raw_detections, frame_w, frame_h
            )

            output: List[Dict] = []
            for det in final_detections:
                if det["cls_int"] == -1:
                    continue
                output.append({
                    "cls": det["cls"],
                    "landing_status": det["landing_status"],
                    "top_left_x": det["top_left_x"],
                    "top_left_y": det["top_left_y"],
                    "bottom_right_x": det["bottom_right_x"],
                    "bottom_right_y": det["bottom_right_y"],
                    "confidence": det["confidence"],
                })

            if Settings.DEBUG:
                cls_counts = Counter(d["cls"] for d in output)
                self.log.debug(
                    f"Tespit: {len(output)} nesne "
                    f"(Taşıt: {cls_counts.get('0', 0)}, "
                    f"İnsan: {cls_counts.get('1', 0)}, "
                    f"UAP: {cls_counts.get('2', 0)}, "
                    f"UAİ: {cls_counts.get('3', 0)})"
                )

            self._frame_count += 1

            return output

        except (SystemExit, KeyboardInterrupt):
            raise
        except torch.cuda.OutOfMemoryError:
            self.log.error("GPU OOM hatası! Inference başarısız.")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return []
        except Exception as e:
            self.log.error(f"Tespit hatası: {e}")
            return []

    def _standard_inference(self, frame: np.ndarray) -> List[Dict]:
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

    def _sahi_detect(self, frame: np.ndarray) -> List[Dict]:
        # Full-frame + parçalı inference birleştir, NMS ile duplikasyonu temizle
        all_detections: List[Dict] = []
        full_dets = self._standard_inference(frame)
        all_detections.extend(full_dets)
        slice_dets = self._sliced_inference(frame)
        all_detections.extend(slice_dets)
        if len(all_detections) > 0:
            all_detections = self._merge_detections_nms(all_detections)

        return all_detections

    def _sliced_inference(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        slice_size = Settings.SAHI_SLICE_SIZE
        overlap = Settings.SAHI_OVERLAP_RATIO
        step = int(slice_size * (1 - overlap))

        all_slice_dets: List[Dict] = []

        with torch.no_grad():
            for y_start in range(0, h, step):
                for x_start in range(0, w, step):
                    x_end = min(x_start + slice_size, w)
                    y_end = min(y_start + slice_size, h)

                    if (x_end - x_start) < slice_size // 2 or \
                       (y_end - y_start) < slice_size // 2:
                        continue

                    tile = frame[y_start:y_end, x_start:x_end]

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

    def _parse_results(self, results) -> List[Dict]:
        detections: List[Dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                model_cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = (
                    float(coords[0]), float(coords[1]),
                    float(coords[2]), float(coords[3]),
                )

                tf_id = self._map_model_class_to_teknofest(model_cls_id)

                detections.append({
                    "cls_int": tf_id,
                    "cls": str(tf_id),
                    "source_cls_id": model_cls_id,
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
        if not detections:
            return detections

        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

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

            nms_keep = ObjectDetector._nms_greedy(
                cls_boxes, cls_scores, Settings.SAHI_MERGE_IOU
            )
            nms_indices = cls_indices[nms_keep]
            keep_indices.extend(nms_indices.tolist())

        nms_results = [detections[i] for i in keep_indices]

        return ObjectDetector._suppress_contained(nms_results)

    @staticmethod
    def _suppress_contained(detections: List[Dict], threshold: float = 0.85) -> List[Dict]:
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        areas = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
        cls_ints = np.array([d["cls_int"] for d in detections])

        order = np.argsort(areas, kind="stable")[::-1]
        keep: List[int] = []
        is_suppressed = np.zeros(len(detections), dtype=bool)
        landing_zone_ids = {Settings.CLASS_UAP, Settings.CLASS_UAI}

        for i_idx, i in enumerate(order):
            if is_suppressed[i]:
                continue
            keep.append(int(i))

            remaining_indices = order[i_idx + 1:]
            valid_mask = ~is_suppressed[remaining_indices] & (cls_ints[remaining_indices] == cls_ints[i])
            valid_remaining = remaining_indices[valid_mask]

            if len(valid_remaining) == 0:
                continue

            box_a = boxes[i]
            boxes_b = boxes[valid_remaining]
            areas_b = areas[valid_remaining]

            inter_x1 = np.maximum(box_a[0], boxes_b[:, 0])
            inter_y1 = np.maximum(box_a[1], boxes_b[:, 1])
            inter_x2 = np.minimum(box_a[2], boxes_b[:, 2])
            inter_y2 = np.minimum(box_a[3], boxes_b[:, 3])

            inter_w = np.maximum(0.0, inter_x2 - inter_x1)
            inter_h = np.maximum(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            ios = inter_area / areas_b

            effective_threshold = (
                Settings.LANDING_ZONE_CONTAINMENT_IOU
                if cls_ints[i] in landing_zone_ids
                else threshold
            )

            suppress_mask = np.zeros(len(valid_remaining), dtype=bool)

            for idx_b, b_idx in enumerate(valid_remaining):
                if ios[idx_b] > effective_threshold:
                    cls_a = cls_ints[i]
                    cls_b = cls_ints[b_idx]

                    if cls_a in landing_zone_ids and cls_b not in landing_zone_ids:
                        continue

                    suppress_mask[idx_b] = True

            is_suppressed[valid_remaining[suppress_mask]] = True

        return [detections[i] for i in keep]

    @staticmethod
    def _nms_greedy(
        boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> List[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum((x2 - x1) * (y2 - y1), 1e-6)

        order = np.argsort(scores, kind="stable")[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / np.maximum(union, 1e-6)

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        result = frame
        if self._clahe is not None:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            l_enhanced = self._clahe.apply(l_channel)

            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=2.0)
        result = cv2.addWeighted(result, 1.3, blurred, -0.3, 0)

        return result

    @staticmethod
    def _post_filter(detections: List[Dict]) -> List[Dict]:
        filtered: List[Dict] = []
        min_size = Settings.MIN_BBOX_SIZE

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1

            if w < min_size or h < min_size:
                continue

            if w > Settings.MAX_BBOX_SIZE or h > Settings.MAX_BBOX_SIZE:
                continue

            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 4.5:
                continue

            filtered.append(det)

        return filtered

    @staticmethod
    def _bbox_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = max(area_a + area_b - inter_area, 1e-6)
        return inter_area / union

    # =========================================================================
    #  İNİŞ UYGUNLUĞU (LANDING STATUS) MANTIĞI
    # =========================================================================

    def _determine_landing_status(
        self,
        detections: List[Dict],
        frame_w: int,
        frame_h: int,
    ) -> List[Dict]:
        """Şartname 4.6: Taşıt/İnsan=-1, UAP/UAİ için kenar/engel/proximity kontrolü → 0 veya 1"""
        # UAP/UAİ dışındaki tüm nesneler engel; IoU yerine intersection/landing_area kullanılır
        # Şartname: "tespit edilen veya tespit edilemeyen herhangi bir nesne"
        # UNKNOWN_OBJECTS_AS_OBSTACLES şartname gereği her zaman True kabul edilir.
        landing_zone_ids = (Settings.CLASS_UAP, Settings.CLASS_UAI)
        obstacles: List[Tuple[float, float, float, float]] = []
        for det in detections:
            if det["cls_int"] not in landing_zone_ids:
                obstacles.append(det["bbox"])

        for det in detections:
            cls_id = det["cls_int"]

            if cls_id in (Settings.CLASS_TASIT, Settings.CLASS_INSAN):
                det["landing_status"] = Settings.LANDING_NOT_AREA

            elif cls_id in landing_zone_ids:
                bbox = det["bbox"]

                # (a) Alan kadrajın kenarına değiyor mu? EDGE_MARGIN_RATIO ile çözünürlüğe orantılı.
                if self._is_touching_edge(bbox, frame_w, frame_h):
                    det["landing_status"] = Settings.LANDING_NOT_SUITABLE
                    self.log.debug("  UAP/UAİ kenar temas → uygun değil")
                    continue

                # (b) Doğrudan engel örtüşme kontrolü
                has_obstacle = self._check_obstacle_overlap(
                    bbox, obstacles, Settings.LANDING_IOU_THRESHOLD
                )

                # (c) Perspektif proximity kontrolü — genişletilmiş bbox
                # Şartname 4.6: "Çekim açısına bağlı yanıltıcı durumda
                # alana yakın cisim alanda gibi görünebilir"
                if not has_obstacle and Settings.LANDING_PROXIMITY_MARGIN > 0:
                    expanded = self._expand_bbox(
                        bbox, Settings.LANDING_PROXIMITY_MARGIN
                    )
                    has_obstacle = self._check_obstacle_overlap(
                        expanded, obstacles, Settings.LANDING_IOU_THRESHOLD
                    )
                    if has_obstacle:
                        self.log.debug(
                            "  UAP/UAİ perspektif proximity engeli → uygun değil"
                        )

                if has_obstacle:
                    det["landing_status"] = Settings.LANDING_NOT_SUITABLE
                else:
                    # (d) Alan tamamen kadrajda, engelsiz, proximity temiz → uygun
                    det["landing_status"] = Settings.LANDING_SUITABLE
                    self.log.debug("  UAP/UAİ → iniş uygun ✓")

            else:
                det["landing_status"] = Settings.LANDING_NOT_AREA

        return detections

    # =========================================================================
    #  ENGEL ÖRTÜŞME KONTROL YARDIMCILARI
    # =========================================================================

    @staticmethod
    def _check_obstacle_overlap(
        landing_bbox: Tuple[float, float, float, float],
        obstacles: List[Tuple[float, float, float, float]],
        threshold: float,
    ) -> bool:
        """
        İniş alanı ile engel listesi arasında eşik üstü kesişim olup olmadığını kontrol eder.

        Args:
            landing_bbox: İniş alanı (x1, y1, x2, y2).
            obstacles: Engel bbox listesi.
            threshold: Minimum overlap oranı (0.0 = herhangi bir piksel).

        Returns:
            Engel bulunduysa True.
        """
        for obs_bbox in obstacles:
            overlap = ObjectDetector._intersection_over_area(
                landing_bbox, obs_bbox
            )
            if overlap > threshold:
                return True
        return False

    @staticmethod
    def _expand_bbox(
        bbox: Tuple[float, float, float, float],
        margin_ratio: float,
    ) -> Tuple[float, float, float, float]:
        """
        Bounding box'ı her yönde margin_ratio oranında genişletir.

        Perspektif kaynaklı yanılgıları yakalamak için iniş alanı
        bbox'ı genişletilerek yakın nesneler kontrol edilir.

        Args:
            bbox: Orijinal (x1, y1, x2, y2).
            margin_ratio: Genişletme oranı (0.15 = %15 her yönde).

        Returns:
            Genişletilmiş (x1, y1, x2, y2).
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        dx = w * margin_ratio
        dy = h * margin_ratio
        return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

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
        if inter_area <= 0:
            return 0.0

        # İniş alanının yüzölçümü
        landing_w = max(0.0, landing_box[2] - landing_box[0])
        landing_h = max(0.0, landing_box[3] - landing_box[1])
        landing_area = landing_w * landing_h

        if landing_area <= 0:
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
    ) -> bool:
        """
        Bounding box'ın kadrajın kenarına değip değmediğini kontrol eder.

        Margin, EDGE_MARGIN_RATIO ile çözünürlüğe orantılı hesaplanır.
        1920px'de ~8px, 3840px'de ~15px — çözünürlük bağımsız davranış.

        Şartnameye göre: UAP/UAİ alanının tamamı kadraj içinde olmalıdır,
        aksi halde iniş durumu "uygun" olamaz.

        Args:
            bbox: (x1, y1, x2, y2) formatında bounding box.
            frame_w: Görüntü genişliği.
            frame_h: Görüntü yüksekliği.

        Returns:
            Kenarına değiyorsa True.
        """
        margin_x = max(1, int(frame_w * Settings.EDGE_MARGIN_RATIO))
        margin_y = max(1, int(frame_h * Settings.EDGE_MARGIN_RATIO))

        x1, y1, x2, y2 = bbox
        return (
            x1 <= margin_x
            or y1 <= margin_y
            or x2 >= (frame_w - margin_x)
            or y2 >= (frame_h - margin_y)
        )
