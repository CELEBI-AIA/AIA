"""YOLOv8 nesne tespiti + iniş uygunluğu (UAP/UAİ).
Model COCO/VisDrone vb. eğitilmiş olabilir; sınıflar TEKNOFEST (0,1,2,3) formatına map edilir."""

import logging
import os
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import Settings
from src.class_contract import CompetitionClassContract
from src.utils import Logger


class ObjectDetector:
    """YOLOv8 tespit, TEKNOFEST sınıf eşlemesi ve iniş uygunluğu."""

    def __init__(self) -> None:
        self.log = Logger("Detector")
        self._frame_count: int = 0
        self._trace_seq: int = 0
        self._last_guardrail_stats: Dict[str, int] = {}
        self._temporal_filter: Optional[Any] = None
        self._use_half: bool = False
        self._class_map_mode: str = "unknown"
        self._model_class_map: Dict[int, int] = {}
        self._uap_uai_model_class_ids: List[int] = []
        self._uap_uai_absent_streak: int = 0

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

        uap_uai_conf = getattr(Settings, "CONFIDENCE_THRESHOLD_UAP_UAI", None)
        if uap_uai_conf is not None:
            self.log.info(
                f"UAP/UAİ conf eşiği: {uap_uai_conf} (Taşıt/İnsan: {Settings.CONFIDENCE_THRESHOLD})"
            )

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
                        classes=None,
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
        return CompetitionClassContract.resolve_alias(normalized)

    def _configure_class_mapping(self) -> None:
        """Model sınıf isimlerine göre COCO/VisDrone/resmi eşleme seçer."""
        CompetitionClassContract.validate_settings_contract()
        custom_map = getattr(Settings, "CUSTOM_CLASS_MAP", None)
        if custom_map is not None and isinstance(custom_map, dict):
            self._class_map_mode = "custom"
            self._model_class_map = {int(k): int(v) for k, v in custom_map.items()}
            self.log.info("CUSTOM_CLASS_MAP kullanılıyor (config override)")
            items = list(self._model_class_map.items())
            items.sort(key=lambda x: x[0])
        else:
            names_raw = getattr(self.model, "names", {})
            items = []  # type: List[Tuple[int, str]]
            if isinstance(names_raw, dict):
                items = [(int(k), str(v)) for k, v in names_raw.items()]
            elif isinstance(names_raw, list):
                items = [(i, str(v)) for i, v in enumerate(names_raw)]

            self.log.info(f"Model sınıfları (model.names): {dict(items) if items else 'boş'}")
            names_for_validation = dict(items)
            if names_for_validation:
                CompetitionClassContract.validate_model_class_order(names_for_validation)

            name_based_map: Dict[int, int] = {}
            normalized_names: Dict[int, str] = {}
            for idx, label in items:
                normalized = self._normalize_label(str(label))
                normalized_names[idx] = normalized
                mapped = self._map_label_to_teknofest_id(str(label))
                if mapped != -1:
                    name_based_map[idx] = mapped

            direct_ids = [0, 1, 2, 3]
            direct_ok = all(name_based_map.get(i, -1) == i for i in direct_ids)
            num_classes = len(items)
            has_uap_uai_in_name_based = (
                Settings.CLASS_UAP in name_based_map.values()
                or Settings.CLASS_UAI in name_based_map.values()
            )
            looks_like_coco = (
                num_classes >= 3
                and normalized_names.get(0) == "person"
                and normalized_names.get(1) == "bicycle"
                and normalized_names.get(2) == "car"
                and not (num_classes <= 10 and has_uap_uai_in_name_based)
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

        log_items: List[Tuple[int, str]] = []
        for idx in sorted(self._model_class_map.keys()):
            label_str = ""
            if hasattr(self.model, "names"):
                names = getattr(self.model, "names", {})
                if isinstance(names, dict):
                    label_str = str(names.get(idx, idx))
                elif isinstance(names, (list, tuple)) and 0 <= idx < len(names):
                    label_str = str(names[idx])
            if not label_str:
                label_str = str(idx)
            log_items.append((idx, label_str))

        self.log.info(
            f"Sınıf eşleme modu: {self._class_map_mode} "
            f"(tanınan model sınıfı: {len(self._model_class_map)})"
        )
        for idx, label in log_items:
            tf_id = self._model_class_map.get(idx, -1)
            tf_name = CompetitionClassContract.display_name(tf_id)
            marker = "✓" if tf_id != -1 else "✗"
            self.log.info(
                f"  {marker} Model sınıf {idx}: '{label}' → TEKNOFEST {tf_id} ({tf_name})"
            )
        has_uap = Settings.CLASS_UAP in self._model_class_map.values()
        has_uai = Settings.CLASS_UAI in self._model_class_map.values()
        if not has_uap:
            self.log.warn("⚠ Model UAP (class 2) sınıfı İÇERMİYOR — UAP tespiti yapılamaz!")
        if not has_uai:
            self.log.warn("⚠ Model UAİ (class 3) sınıfı İÇERMİYOR — UAİ tespiti yapılamaz!")

        self._uap_uai_model_class_ids = sorted(
            model_cls
            for model_cls, tf_cls in self._model_class_map.items()
            if tf_cls in (Settings.CLASS_UAP, Settings.CLASS_UAI)
        )
        if self._uap_uai_model_class_ids:
            self.log.info(
                "UAP/UAİ focused-pass model class ids: "
                f"{self._uap_uai_model_class_ids}"
            )

    def _map_model_class_to_teknofest(self, model_cls_id: int) -> int:
        return self._model_class_map.get(model_cls_id, -1)

    @staticmethod
    def _resolve_nms_mode() -> str:
        mode = str(getattr(Settings, "NMS_MODE", "")).strip().lower()
        if mode in {"class_aware", "agnostic", "hybrid"}:
            return mode
        return "agnostic" if bool(getattr(Settings, "AGNOSTIC_NMS", False)) else "class_aware"

    def _build_inference_config(self, runtime_profile: str) -> Dict[str, Any]:
        profile = str(runtime_profile or "default").strip().lower()
        if profile == "light":
            return {
                "imgsz": max(
                    256,
                    int(
                        getattr(
                            Settings,
                            "LIGHT_PROFILE_INFERENCE_SIZE",
                            Settings.INFERENCE_SIZE,
                        )
                    ),
                ),
                "conf": float(
                    max(
                        Settings.CONFIDENCE_THRESHOLD,
                        getattr(
                            Settings,
                            "LIGHT_PROFILE_CONFIDENCE_THRESHOLD",
                            Settings.CONFIDENCE_THRESHOLD,
                        ),
                    )
                ),
                "iou": float(Settings.NMS_IOU_THRESHOLD),
                "max_det": max(
                    1,
                    int(
                        getattr(
                            Settings,
                            "LIGHT_PROFILE_MAX_DETECTIONS",
                            Settings.MAX_DETECTIONS,
                        )
                    ),
                ),
                "augment": bool(
                    getattr(Settings, "LIGHT_PROFILE_AUGMENTED_INFERENCE", False)
                ),
                "sahi_enabled": bool(
                    getattr(Settings, "LIGHT_PROFILE_SAHI_ENABLED", False)
                ),
                "merge_iou": float(Settings.SAHI_MERGE_IOU),
                "hybrid_iou": float(
                    getattr(Settings, "HYBRID_NMS_IOU_THRESHOLD", 0.65)
                ),
            }

        conf_global = float(Settings.CONFIDENCE_THRESHOLD)
        conf_uap_uai = getattr(Settings, "CONFIDENCE_THRESHOLD_UAP_UAI", None)
        conf_infer = min(conf_global, conf_uap_uai) if conf_uap_uai is not None else conf_global
        return {
            "imgsz": int(Settings.INFERENCE_SIZE),
            "conf": float(conf_infer),
            "iou": float(Settings.NMS_IOU_THRESHOLD),
            "max_det": int(Settings.MAX_DETECTIONS),
            "augment": bool(Settings.AUGMENTED_INFERENCE),
            "sahi_enabled": bool(Settings.SAHI_ENABLED),
            "merge_iou": float(Settings.SAHI_MERGE_IOU),
            "hybrid_iou": float(getattr(Settings, "HYBRID_NMS_IOU_THRESHOLD", 0.65)),
        }

    def detect(self, frame: np.ndarray, runtime_profile: str = "default", **kwargs) -> List[Dict]:
        try:
            inference_cfg = self._build_inference_config(runtime_profile)
            processed = self._preprocess(frame)
            stage_trace: List[Dict[str, Any]] = []
            if inference_cfg["sahi_enabled"]:
                raw_detections = self._sahi_detect(processed, inference_cfg=inference_cfg)
            else:
                raw_detections = self._standard_inference(
                    processed, inference_cfg=inference_cfg
                )
            focused_dets = self._focused_uap_uai_inference(
                processed, inference_cfg=inference_cfg
            )
            if focused_dets:
                raw_detections.extend(focused_dets)
            self._collect_stage_stats(stage_trace, "raw_model_output", raw_detections)
            self._track_uap_uai_absence(raw_detections)

            raw_detections = self._filter_by_confidence(raw_detections)
            self._collect_stage_stats(stage_trace, "confidence_filter", raw_detections)

            nms_mode = self._resolve_nms_mode()
            if nms_mode == "agnostic":
                self._log_nms_mode_comparison(raw_detections, inference_cfg)
            raw_detections = self._apply_runtime_nms(
                raw_detections, inference_cfg=inference_cfg
            )
            self._collect_stage_stats(stage_trace, f"nms_{nms_mode}", raw_detections)
            raw_detections = self._suppress_landing_zone_class_conflicts(raw_detections)
            self._collect_stage_stats(stage_trace, "uap_uai_conflict_suppress", raw_detections)
            raw_detections = self._post_filter(raw_detections, altitude=kwargs.get("altitude"))
            self._collect_stage_stats(stage_trace, "min_size_post_filter", raw_detections)
            
            try:
                from src.postprocess import apply_guardrails
                raw_detections, self._last_guardrail_stats = apply_guardrails(raw_detections)
            except ImportError:
                self._last_guardrail_stats = {}
            self._collect_stage_stats(stage_trace, "guardrails", raw_detections)

            if getattr(Settings, "TEMPORAL_FILTER_ENABLED", True):
                try:
                    from src.temporal_filter import TemporalConsistencyFilter
                    if self._temporal_filter is None:
                        self._temporal_filter = TemporalConsistencyFilter()
                    raw_detections = self._temporal_filter.filter(raw_detections)
                except ImportError:
                    pass
            self._collect_stage_stats(stage_trace, "temporal_filter", raw_detections)

            frame_h, frame_w = frame.shape[:2]
            try:
                from src.uap_uai import determine_landing_status
                final_detections = determine_landing_status(
                    raw_detections, frame_w, frame_h, frame
                )
            except ImportError:
                final_detections = raw_detections
            self._collect_stage_stats(stage_trace, "landing_status", final_detections)

            output: List[Dict] = []
            conf_tasit_insan = float(Settings.CONFIDENCE_THRESHOLD)
            for det in final_detections:
                if det["cls_int"] == -1:
                    continue
                if det["cls_int"] in (Settings.CLASS_TASIT, Settings.CLASS_INSAN):
                    if det["confidence"] < conf_tasit_insan:
                        continue
                # Şartname: motion_status → Taşıt(0)=0/1, İnsan(1)=-1, UAP(2)=-1, UAİ(3)=-1
                # Varsayılan -1; MovementEstimator.annotate() taşıtlar için sonra günceller
                cls_id = det["cls_int"]
                default_motion = "-1"
                if cls_id in (Settings.CLASS_UAP, Settings.CLASS_UAI):
                    landing_status = str(det.get("landing_status", "0"))
                    if landing_status not in {"0", "1"}:
                        landing_status = "0"
                else:
                    landing_status = "-1"
                if cls_id in (Settings.CLASS_UAP, Settings.CLASS_UAI) and "landing_status" not in det:
                    self.log.warn(
                        "UAP/UAİ detection missing landing_status; defaulting to 0"
                    )
                output.append({
                    "cls": det["cls"],
                    "landing_status": landing_status,
                    "motion_status": default_motion,
                    "top_left_x": det["top_left_x"],
                    "top_left_y": det["top_left_y"],
                    "bottom_right_x": det["bottom_right_x"],
                    "bottom_right_y": det["bottom_right_y"],
                    "confidence": det["confidence"],
                    "trace_id": det.get("trace_id", ""),
                })

            self._collect_stage_stats(stage_trace, "final_json_candidates", output)
            self._log_stage_trace(stage_trace)

            if Settings.DEBUG:
                cls_counts = Counter(d["cls"] for d in output)
                self.log.debug(
                    f"Tespit: {len(output)} nesne "
                    f"(Taşıt: {cls_counts.get('0', 0)}, "
                    f"İnsan: {cls_counts.get('1', 0)}, "
                    f"UAP: {cls_counts.get('2', 0)}, "
                    f"UAİ: {cls_counts.get('3', 0)})"
                )

            if Settings.DEBUG:
                self.log.debug(
                    f"Inference profile={runtime_profile} "
                    f"sahi={'on' if inference_cfg['sahi_enabled'] else 'off'} "
                    f"imgsz={inference_cfg['imgsz']} conf={inference_cfg['conf']:.2f}"
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

    def _standard_inference(
        self,
        frame: np.ndarray,
        inference_cfg: Dict[str, Any],
    ) -> List[Dict]:
        with torch.no_grad():
            results = self.model.predict(
                source=frame,
                imgsz=int(inference_cfg["imgsz"]),
                conf=float(inference_cfg["conf"]),
                iou=float(inference_cfg["iou"]),
                classes=None,
                device=self.device,
                verbose=False,
                save=False,
                half=self._use_half,
                agnostic_nms=self._resolve_nms_mode() == "agnostic",
                max_det=int(inference_cfg["max_det"]),
                augment=bool(inference_cfg["augment"]),
            )
        return self._parse_results(results)

    def _focused_uap_uai_inference(
        self,
        frame: np.ndarray,
        inference_cfg: Dict[str, Any],
    ) -> List[Dict]:
        if not bool(getattr(Settings, "UAP_UAI_FOCUSED_PASS_ENABLED", False)):
            return []
        if not self._uap_uai_model_class_ids:
            return []

        interval = max(1, int(getattr(Settings, "UAP_UAI_FOCUSED_PASS_INTERVAL", 2)))
        if self._frame_count % interval != 0:
            return []

        focus_conf = float(
            getattr(
                Settings,
                "UAP_UAI_FOCUSED_PASS_CONF",
                getattr(Settings, "CONFIDENCE_THRESHOLD_UAP_UAI", Settings.CONFIDENCE_THRESHOLD),
            )
        )
        focus_conf = max(0.001, min(1.0, focus_conf))
        focus_imgsz = max(
            256,
            int(getattr(Settings, "UAP_UAI_FOCUSED_PASS_IMG_SIZE", inference_cfg["imgsz"])),
        )

        with torch.no_grad():
            results = self.model.predict(
                source=frame,
                imgsz=focus_imgsz,
                conf=focus_conf,
                iou=float(inference_cfg["iou"]),
                device=self.device,
                verbose=False,
                save=False,
                half=self._use_half,
                classes=list(self._uap_uai_model_class_ids),
                agnostic_nms=False,
                max_det=int(inference_cfg["max_det"]),
                augment=False,
            )
        focused = self._parse_results(results)
        if bool(getattr(Settings, "DEBUG", False)) and focused:
            cls_counts = Counter(str(d.get("cls", "")) for d in focused)
            self.log.debug(
                "FocusedPass(UAP/UAİ) "
                f"total={len(focused)} uap={cls_counts.get('2', 0)} "
                f"uai={cls_counts.get('3', 0)} conf={focus_conf:.2f} imgsz={focus_imgsz}"
            )
        return focused

    def _sahi_detect(
        self,
        frame: np.ndarray,
        inference_cfg: Dict[str, Any],
    ) -> List[Dict]:
        # Full-frame + parçalı inference birleştir, NMS ile duplikasyonu temizle
        all_detections: List[Dict] = []
        full_dets = self._standard_inference(frame, inference_cfg=inference_cfg)
        all_detections.extend(full_dets)
        slice_dets = self._sliced_inference(frame, inference_cfg=inference_cfg)
        all_detections.extend(slice_dets)
        return all_detections

    def _sliced_inference(
        self,
        frame: np.ndarray,
        inference_cfg: Dict[str, Any],
    ) -> List[Dict]:
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
                        conf=float(inference_cfg["conf"]),
                        iou=float(inference_cfg["iou"]),
                        classes=None,
                        device=self.device,
                        verbose=False,
                        save=False,
                        half=self._use_half,
                        agnostic_nms=self._resolve_nms_mode() == "agnostic",
                        max_det=int(inference_cfg["max_det"]),
                        augment=bool(inference_cfg["augment"]),
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
                    "trace_id": self._next_trace_id(),
                    "cls_int": tf_id,
                    "cls": str(tf_id),
                    "class_label": CompetitionClassContract.display_name(tf_id),
                    "source_cls_id": model_cls_id,
                    "confidence": int(conf * 10000) / 10000,
                    "top_left_x": round(x1, 2),
                    "top_left_y": round(y1, 2),
                    "bottom_right_x": round(x2, 2),
                    "bottom_right_y": round(y2, 2),
                    "bbox": (x1, y1, x2, y2),
                })
        return detections

    def _next_trace_id(self) -> str:
        self._trace_seq += 1
        return f"f{self._frame_count:06d}-d{self._trace_seq:08d}"

    @staticmethod
    def _filter_by_confidence(detections: List[Dict]) -> List[Dict]:
        conf_global = float(Settings.CONFIDENCE_THRESHOLD)
        conf_uap_uai = getattr(Settings, "CONFIDENCE_THRESHOLD_UAP_UAI", None)
        if conf_uap_uai is None:
            conf_uap_uai = conf_global
        conf_uap_uai = float(conf_uap_uai)

        filtered: List[Dict] = []
        for det in detections:
            cls_int = int(det.get("cls_int", -1))
            conf = float(det.get("confidence", 0.0))
            threshold = conf_uap_uai if cls_int in (Settings.CLASS_UAP, Settings.CLASS_UAI) else conf_global
            if conf >= threshold:
                filtered.append(det)
        return filtered

    def _track_uap_uai_absence(self, detections: List[Dict]) -> None:
        has_uap_uai = any(
            int(det.get("cls_int", -1)) in (Settings.CLASS_UAP, Settings.CLASS_UAI)
            for det in detections
        )
        if has_uap_uai:
            self._uap_uai_absent_streak = 0
            return

        self._uap_uai_absent_streak += 1
        if self._uap_uai_absent_streak in (120, 360):
            self.log.warn(
                "Ham model çıktısında uzun süredir UAP/UAİ yok "
                f"(streak={self._uap_uai_absent_streak}). "
                "Veri sekansında UAP/UAİ bulunmuyor olabilir veya model eşiği yüksek kalıyor olabilir."
            )

    def _collect_stage_stats(
        self,
        stage_trace: List[Dict[str, Any]],
        stage: str,
        detections: List[Dict],
    ) -> None:
        counts = Counter(str(det.get("cls", "")) for det in detections)
        stage_trace.append(
            {
                "stage": stage,
                "total": len(detections),
                "uap": int(counts.get("2", 0)),
                "uai": int(counts.get("3", 0)),
            }
        )

    def _log_stage_trace(self, stage_trace: List[Dict[str, Any]]) -> None:
        if not bool(getattr(Settings, "DEBUG", False)) or not stage_trace:
            return
        frame_token = f"f{self._frame_count:06d}"
        parts: List[str] = []
        previous = None
        for row in stage_trace:
            if previous is None:
                delta_total = 0
                delta_uap = 0
                delta_uai = 0
            else:
                delta_total = previous["total"] - row["total"]
                delta_uap = previous["uap"] - row["uap"]
                delta_uai = previous["uai"] - row["uai"]
            parts.append(
                f"{row['stage']} total={row['total']} (drop={delta_total}), "
                f"uap={row['uap']} (drop={delta_uap}), uai={row['uai']} (drop={delta_uai})"
            )
            previous = row
        self.log.debug(f"PipelineTrace[{frame_token}] " + " | ".join(parts))

    def _log_nms_mode_comparison(
        self,
        detections: List[Dict],
        inference_cfg: Dict[str, Any],
    ) -> None:
        if not bool(getattr(Settings, "DEBUG", False)):
            return
        class_aware = self._merge_detections_nms(detections)
        agnostic = self._merge_detections_nms_agnostic(
            detections,
            iou_threshold=float(inference_cfg["merge_iou"]),
        )
        aware_ids = {d.get("trace_id") for d in class_aware}
        agnostic_ids = {d.get("trace_id") for d in agnostic}
        cross_class_drop = len(aware_ids - agnostic_ids)
        self.log.debug(
            "NMSCompare mode=agnostic "
            f"class_aware={len(class_aware)} agnostic={len(agnostic)} "
            f"cross_class_drop={cross_class_drop}"
        )

    def _apply_runtime_nms(
        self,
        detections: List[Dict],
        inference_cfg: Dict[str, Any],
    ) -> List[Dict]:
        if not detections:
            return []

        mode = self._resolve_nms_mode()
        if mode == "agnostic":
            return self._merge_detections_nms_agnostic(
                detections, iou_threshold=float(inference_cfg["merge_iou"])
            )
        if mode == "hybrid":
            class_aware = self._merge_detections_nms(detections)
            return self._merge_detections_nms_agnostic(
                class_aware,
                iou_threshold=float(inference_cfg["hybrid_iou"]),
            )
        return self._merge_detections_nms(detections)

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
    def _merge_detections_nms_agnostic(
        detections: List[Dict],
        iou_threshold: float,
    ) -> List[Dict]:
        if not detections:
            return []
        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)
        keep = ObjectDetector._nms_greedy(boxes, scores, float(iou_threshold))
        return ObjectDetector._suppress_contained([detections[i] for i in keep])

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
        
        # Otonom Adaptif CLAHE ve Termal/RGB kontrolü
        if self._clahe is not None:
            # Check if thermal (single channel disguised as 3 or actually 1)
            is_thermal = False
            if len(frame.shape) == 2:
                is_thermal = True
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # If R, G, B channels are highly correlated/identical, it's likely grayscale/thermal
                b, g, r = cv2.split(frame)
                diff_bg = cv2.absdiff(b, g)
                diff_gr = cv2.absdiff(g, r)
                if cv2.mean(diff_bg)[0] < 2.0 and cv2.mean(diff_gr)[0] < 2.0:
                    is_thermal = True

            mean_brightness = np.mean(frame)
            # Apply CLAHE if it's thermal OR if it's RGB but too dark/foggy (low contrast)
            if is_thermal:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else result
                enhanced = self._clahe.apply(gray)
                result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            elif mean_brightness < 90.0 or mean_brightness > 200.0: # Too dark (night) or too washed out (fog/snow)
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                l_enhanced = self._clahe.apply(l_channel)
                lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
                result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # If RGB and normal brightness, skip CLAHE to save CPU and prevent artifacts

        # Hafif keskinleştirme (bulanklık toleransı - FR-007)
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=2.0)
        result = cv2.addWeighted(result, 1.3, blurred, -0.3, 0)

        return result

    @staticmethod
    def _post_filter(detections: List[Dict], altitude: Optional[float] = None) -> List[Dict]:
        filtered: List[Dict] = []
        class_filters = getattr(Settings, "CLASS_ADAPTIVE_FILTERS", {}) or {}
        default_min_size = max(1, int(Settings.MIN_BBOX_SIZE))
        default_max_size = max(default_min_size, int(getattr(Settings, "MAX_BBOX_SIZE", 9999)))
        default_max_aspect = 4.5
        
        # Scale thresholds based on altitude (reference 50m)
        # Closer to ground (lower altitude) = larger boxes expected
        scale_factor = 1.0
        if altitude is not None and altitude > 1.0:
            ref_altitude = getattr(Settings, "DEFAULT_ALTITUDE", 50.0)
            scale_factor = ref_altitude / max(5.0, altitude)

        post_filter_exempt = frozenset(str(c) for c in getattr(Settings, "GUARDRAIL_EXEMPT_CLASSES", ("2", "3")))
        for det in detections:
            cls_key = str(det.get("cls_int", det.get("cls", "")))
            if cls_key in post_filter_exempt:
                filtered.append(det)
                continue

            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1

            cfg = class_filters.get(cls_key, {})
            base_min = max(1, int(cfg.get("min_size", default_min_size)))
            base_max = max(base_min, int(cfg.get("max_size", default_max_size)))
            min_size = base_min * scale_factor
            default_min_floor = max(1, int(getattr(Settings, "MIN_BBOX_SIZE_FLOOR", 8)))
            min_floor = max(1, int(cfg.get("min_floor", default_min_floor)))
            min_size = max(min_floor, min_size)
            max_size = base_max * scale_factor
            max_aspect = float(cfg.get("max_aspect", default_max_aspect))

            if w < min_size or h < min_size:
                continue
            if w > max_size or h > max_size:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > max_aspect:
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

    def _suppress_landing_zone_class_conflicts(
        self,
        detections: List[Dict],
    ) -> List[Dict]:
        if not detections:
            return []

        threshold = float(
            getattr(Settings, "UAP_UAI_CONFLICT_IOU_THRESHOLD", 0.55)
        )
        if threshold <= 0.0:
            return detections

        landing_zone_ids = {Settings.CLASS_UAP, Settings.CLASS_UAI}
        candidate_indices = [
            idx
            for idx, det in enumerate(detections)
            if int(det.get("cls_int", -1)) in landing_zone_ids
        ]
        if len(candidate_indices) < 2:
            return detections

        def _bbox(det: Dict) -> Tuple[float, float, float, float]:
            if "bbox" in det and len(det["bbox"]) == 4:
                x1, y1, x2, y2 = det["bbox"]
            else:
                x1 = float(det.get("top_left_x", 0.0))
                y1 = float(det.get("top_left_y", 0.0))
                x2 = float(det.get("bottom_right_x", x1 + 1.0))
                y2 = float(det.get("bottom_right_y", y1 + 1.0))
            return (float(x1), float(y1), float(x2), float(y2))

        def _area(box: Tuple[float, float, float, float]) -> float:
            return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

        keep_mask = np.ones(len(detections), dtype=bool)
        ranked_indices = sorted(
            candidate_indices,
            key=lambda idx: (
                float(detections[idx].get("confidence", 0.0)),
                _area(_bbox(detections[idx])),
            ),
            reverse=True,
        )

        suppressed = 0
        for rank_pos, idx in enumerate(ranked_indices):
            if not keep_mask[idx]:
                continue
            det_i = detections[idx]
            cls_i = int(det_i.get("cls_int", -1))
            box_i = _bbox(det_i)
            for jdx in ranked_indices[rank_pos + 1 :]:
                if not keep_mask[jdx]:
                    continue
                det_j = detections[jdx]
                cls_j = int(det_j.get("cls_int", -1))
                if cls_i == cls_j:
                    continue
                if self._bbox_iou(box_i, _bbox(det_j)) >= threshold:
                    keep_mask[jdx] = False
                    suppressed += 1

        if suppressed > 0 and bool(getattr(Settings, "DEBUG", False)):
            self.log.debug(
                "LandingZoneConflictSuppress "
                f"removed={suppressed} iou_threshold={threshold:.2f}"
            )

        return [det for idx, det in enumerate(detections) if bool(keep_mask[idx])]

    # =========================================================================



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
