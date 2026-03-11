"""Veri seti yükleyici. datasets/ klasörünü recursive tarar, uzantıya göre görüntü bulur.
VID: aynı klasördeki sıralı kareler. DET: tüm görüntülerden rastgele örnek."""

import os
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Any, Optional

import cv2

from config.settings import Settings
from src.utils import Logger


def _collect_images_recursive(root: str) -> List[str]:
    """datasets/ altında recursive tara, sadece uzantıya göre eşleşen dosyaları topla."""
    exts = tuple(e.lower() for e in Settings.IMAGE_EXTENSIONS)
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(exts):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def _collect_videos_recursive(root: str) -> List[str]:
    """datasets/ altında recursive tara, sadece uzantıya göre eşleşen videoları topla."""
    exts = tuple(e.lower() for e in Settings.VIDEO_EXTENSIONS)
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(exts):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def _group_by_directory(paths: List[str]) -> Dict[str, List[str]]:
    """Dosyaları üst klasörlerine göre grupla (VID sekans seçimi için)."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        parent = os.path.dirname(p)
        groups[parent].append(p)
    return dict(groups)


def get_available_sequences() -> Dict[str, Dict[str, Any]]:
    """Scan datasets/ for image sequences (folders with >=2 images) and video files."""
    datasets_dir = Settings.DATASETS_DIR
    if not os.path.isdir(datasets_dir):
        return {}

    img_exts = tuple(e.lower() for e in Settings.IMAGE_EXTENSIONS)
    vid_exts = tuple(e.lower() for e in Settings.VIDEO_EXTENSIONS)

    all_images = []
    videos = []

    for dirpath, _, filenames in os.walk(datasets_dir):
        for f in filenames:
            ext = f.lower()
            if ext.endswith(img_exts):
                all_images.append(os.path.join(dirpath, f))
            elif ext.endswith(vid_exts):
                videos.append(os.path.join(dirpath, f))

    groups = _group_by_directory(all_images)
    sequences = {}

    for k, v in groups.items():
        if len(v) >= 2:
            rel_key = os.path.relpath(k, datasets_dir).replace(os.sep, "/")
            name = os.path.basename(k)
            sequence_key = f"img:{rel_key}"
            sequences[sequence_key] = {
                "type": "image_sequence",
                "path": k,
                "count": len(v),
                "files": sorted(v),
                "label": name,
                "aliases": [name],
            }

    for v in videos:
        rel_key = os.path.relpath(v, datasets_dir).replace(os.sep, "/")
        name = os.path.basename(v)
        sequence_key = f"vid:{rel_key}"
        sequences[sequence_key] = {
            "type": "video",
            "path": v,
            "count": "vid_file",
            "label": name,
            "aliases": [name],
        }

    return sequences


class DatasetLoader:
    """datasets/ recursive taranır, uzantıya göre (.jpg, .png vb.) tüm görüntüler bulunur."""

    def __init__(self, prefer_vid: bool = True, seed: Optional[int] = None, sequence: Optional[str] = None) -> None:
        self.log = Logger("DataLoader")
        self._frames: List[str] = []  # Paths for image sequences, or empty if video file
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._video_total_frames: int = 0
        self._index: int = 0
        self._mode: str = "unknown"
        self._sequence_name: str = ""

        datasets_dir = Settings.DATASETS_DIR
        if not os.path.isdir(datasets_dir):
            self.log.error(f"Veri seti dizini bulunamadı: {datasets_dir}")
            self.log.error("  → 'datasets/' klasörünü oluşturup görüntü dosyalarını koyun.")
            return

        self.log.info(f"Veri seti taranıyor (recursive): {datasets_dir}")

        all_images = _collect_images_recursive(datasets_dir)
        all_videos = _collect_videos_recursive(datasets_dir)

        if not all_images and not all_videos:
            self.log.error("Hiçbir görüntü bulunamadı!")
            self.log.error(
                f"  → Desteklenen görüntü uzantıları: {Settings.IMAGE_EXTENSIONS}"
            )
            self.log.error(
                f"  → Desteklenen video uzantıları: {Settings.VIDEO_EXTENSIONS}"
            )
            return

        if all_images:
            self.log.success(f"Toplam {len(all_images)} görüntü bulundu")
        if all_videos:
            self.log.success(f"Toplam {len(all_videos)} video bulundu")

        if seed is not None:
            random.seed(seed)
            self.log.info(f"Deterministik mod: seed={seed}")

        if prefer_vid:
            self._load_video_sequence(all_images, sequence=sequence)
        else:
            self._load_detection_images(all_images)

        if self._frames:
            self.log.success(
                f"Mod: {self._mode.upper()} | "
                f"Toplam kare: {len(self._frames)} | "
                f"{'Sekans: ' + self._sequence_name if self._sequence_name else ''}"
            )

    def _load_video_sequence(self, all_images: List[str], sequence: Optional[str] = None) -> None:
        """Belirtilen dizi adını veya en çok görüntü içeren klasörü sekans olarak seç, veya bir video dosyası aç."""
        available = get_available_sequences()
        
        if not available:
            self.log.warn("Sıralı sekans (görüntü/video) bulunamadı, DET moduna geçiliyor")
            self._load_detection_images(all_images)
            return

        chosen_key = None
        if sequence is not None and sequence in available:
            chosen_key = sequence
        elif sequence is not None:
            alias_matches = []
            for key, sequence_info in available.items():
                aliases = sequence_info.get("aliases", [])
                if sequence in aliases:
                    alias_matches.append(key)

            if len(alias_matches) == 1:
                chosen_key = alias_matches[0]
                self.log.warn(
                    f"Sekans '{sequence}' eski basename alias ile eşleşti, lütfen unique key kullanın: {chosen_key}"
                )
            elif len(alias_matches) > 1:
                self.log.warn(
                    "Birden fazla sekans aynı alias ile eşleşti "
                    f"('{sequence}'): {', '.join(alias_matches)}. Unique key kullanın."
                )

        if chosen_key is None:
            if sequence is not None:
                self.log.warn(f"Sekans '{sequence}' bulunamadı.")

            # Sequence name is not provided or not valid, pick the largest image sequence, or first video
            img_seqs = [(k, v) for k, v in available.items() if v["type"] == "image_sequence"]
            if img_seqs:
                img_seqs.sort(key=lambda x: x[1]["count"], reverse=True)
                chosen_key = img_seqs[0][0]
            else:
                chosen_key = list(available.keys())[0]

        chosen = available[chosen_key]
        self._sequence_name = chosen.get("label", chosen_key)
        self._mode = "vid"

        if chosen["type"] == "video":
            self.log.info(f"Video dosyası açılıyor: {chosen['path']}")
            self._video_capture = cv2.VideoCapture(chosen["path"])
            if not self._video_capture.isOpened():
                self.log.error(f"Video açılamadı: {chosen['path']}")
                self._video_capture = None
                return
            self._video_total_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if self._video_total_frames <= 0:
                self.log.warn(
                    "Video metadata frame count bilgisi yok/bozuk. "
                    "Tutarlılık için kare sayısı manuel olarak hesaplanıyor."
                )
                self._video_total_frames = self._estimate_video_frame_count()
            self.log.info(f"Sekans: {self._sequence_name} (Video: ~{self._video_total_frames} kare)")
        else:
            self._frames = chosen["files"]
            self.log.info(f"Sekans: {self._sequence_name} (Resim dizisi: {len(self._frames)} kare)")

    def _estimate_video_frame_count(self) -> int:
        """Fallback frame-count strategy for videos with missing metadata."""
        if self._video_capture is None:
            return 0

        # Start from the beginning to keep __len__ and iterator semantics aligned.
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        while True:
            ret, _ = self._video_capture.read()
            if not ret:
                break
            frame_count += 1

        # Best-effort rewind for subsequent iterations.
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame_count

    def _load_detection_images(self, all_images: List[str]) -> None:
        """Tüm görüntülerden rastgele örnek al."""
        sample_size = min(Settings.SIMULATION_DET_SAMPLE_SIZE, len(all_images))
        selected = sorted(random.sample(all_images, sample_size))

        self._frames = selected
        self._mode = "det"
        self.log.info(
            f"Rastgele {sample_size} görüntü seçildi (toplam {len(all_images)} içinden)"
        )

    def __len__(self) -> int:
        if self._video_capture is not None:
            return int(max(0, self._video_total_frames))
        return len(self._frames)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Reset iteration cursor.

        For video sources we best-effort seek to frame 0. If seek fails, iteration
        continues from the current decoder position and may stop immediately when
        the stream is already at EOF.
        """
        self._index = 0
        if self._video_capture is not None:
            reset_ok = self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_pos = self._video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            if (not reset_ok) or current_pos > 0.5:
                self.log.warn(
                    "Video akışı başa sarılamadı (CAP_PROP_POS_FRAMES=0). "
                    "Iterator mevcut pozisyondan devam edecek."
                )
        return self

    def __next__(self) -> Dict[str, Any]:
        while True:
            if self._video_capture is not None:
                ret, frame = self._video_capture.read()
                if not ret:
                    raise StopIteration
                frame_path = f"video_frame_{self._index:05d}.jpg"
            else:
                if self._index >= len(self._frames):
                    raise StopIteration
                frame_path = self._frames[self._index]
                frame = cv2.imread(frame_path)

                if frame is None:
                    self.log.warn(f"Görüntü okunamadı, atlanıyor: {frame_path}")
                    self._index += 1
                    continue

            # GPS simülasyonu (şartname 3.2.2): ilk 1 dk sağlıklı, sonra %33 sağlıksız
            if self._mode == "vid":
                if self._index < 450:
                    gps_health = 1
                else:
                    gps_health = 0 if random.random() < 0.33 else 1
            else:
                gps_health = 1

            # Simülasyonda her zaman GPS=0 simüle et (görsel odometri testi)
            if getattr(Settings, "SIMULATION_FORCE_GPS_UNHEALTHY", False):
                gps_health = 0

            result: Dict[str, Any] = {
                "frame": frame,
                "frame_idx": self._index,
                "filename": os.path.basename(frame_path),
                "mode": self._mode,
                "gps_health": gps_health,
                "server_data": {
                    "frame_id": self._index,
                    "gps_health": gps_health,
                    "gps_health_status": gps_health,
                    "translation_x": float(self._index * 0.5) if gps_health == 1 else "NaN",
                    "translation_y": float(self._index * 0.1) if gps_health == 1 else "NaN",
                    "translation_z": Settings.DEFAULT_ALTITUDE if gps_health == 1 else "NaN",
                },
            }

            self._index += 1
            return result

        raise StopIteration

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_ready(self) -> bool:
        if self._video_capture is not None:
            return self._video_capture.isOpened() and len(self) > 0
        return len(self._frames) > 0

    def __del__(self):
        if hasattr(self, "_video_capture") and self._video_capture is not None:
            self._video_capture.release()
