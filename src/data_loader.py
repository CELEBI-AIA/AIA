"""
TEKNOFEST Havacılıkta Yapay Zeka — Veri Seti Yükleyici
=======================================================
VisDrone veri setlerini otomatik keşfeder ve iterator olarak sunar.

Desteklenen veri setleri:
    - VisDrone2019-DET-*  → Tekil fotoğraflar (Görev 1: Nesne Tespiti)
    - VisDrone2019-VID-*  → Video sekansları (Görev 2: Pozisyon Kestirimi)

Kullanım:
    from src.data_loader import DatasetLoader

    loader = DatasetLoader()
    for frame_info in loader:
        frame = frame_info["frame"]       # numpy BGR görüntüsü
        idx   = frame_info["frame_idx"]   # kare numarası
        # ...
"""

import os
import random
from glob import glob
from typing import Dict, Iterator, List, Optional, Any

import cv2
import numpy as np

from config.settings import Settings
from src.utils import Logger


class DatasetLoader:
    """
    VisDrone veri setlerini otomatik keşfeden ve sıralı kare sunan yükleyici.

    Başlatılırken datasets/ klasörünü tarar:
        - VID veri seti bulunursa → rastgele bir sekans seçer (Görev 2: sıralı kareler)
        - DET veri seti bulunursa → rastgele N fotoğraf seçer (Görev 1: tekil kareler)
        - İkisi de varsa → VID tercih edilir (odometri testi daha değerli)

    Iterator protocol destekler: `for frame_info in loader: ...`
    """

    def __init__(self, prefer_vid: bool = True) -> None:
        """
        DatasetLoader'ı başlatır — datasets/ klasörünü tarar.

        Args:
            prefer_vid: True ise VID veri seti öncelikli (Görev 2 testi).
                        False ise DET veri seti öncelikli (Görev 1 testi).
        """
        self.log = Logger("DataLoader")
        self._frames: List[str] = []
        self._index: int = 0
        self._mode: str = "unknown"  # "vid" veya "det"
        self._sequence_name: str = ""

        datasets_dir = Settings.DATASETS_DIR
        if not os.path.isdir(datasets_dir):
            self.log.error(f"Veri seti dizini bulunamadı: {datasets_dir}")
            self.log.error("  → 'datasets/' klasörünü oluşturup VisDrone verilerini koyun.")
            return

        self.log.info(f"Veri seti dizini taranıyor: {datasets_dir}")

        # ---- Veri Seti Keşfi ----
        vid_dir = self._find_dataset(datasets_dir, "VID")
        det_dir = self._find_dataset(datasets_dir, "DET")

        if vid_dir:
            self.log.success(f"VID veri seti bulundu: {os.path.basename(vid_dir)}")
        if det_dir:
            self.log.success(f"DET veri seti bulundu: {os.path.basename(det_dir)}")

        if not vid_dir and not det_dir:
            self.log.error("Hiçbir VisDrone veri seti bulunamadı!")
            self.log.error("  → datasets/ klasörüne VisDrone2019-DET-* veya VID-* koyun.")
            return

        # ---- Mod Seçimi ----
        if prefer_vid and vid_dir:
            self._load_video_sequence(vid_dir)
        elif det_dir:
            self._load_detection_images(det_dir)
        elif vid_dir:
            self._load_video_sequence(vid_dir)

        if self._frames:
            self.log.success(
                f"Mod: {self._mode.upper()} | "
                f"Toplam kare: {len(self._frames)} | "
                f"{'Sekans: ' + self._sequence_name if self._sequence_name else ''}"
            )

    # =========================================================================
    #  VERİ SETİ KEŞFİ
    # =========================================================================

    @staticmethod
    def _find_dataset(datasets_dir: str, dataset_type: str) -> Optional[str]:
        """
        datasets/ içinde VisDrone veri setini arar.

        Args:
            datasets_dir: Ana veri seti dizini.
            dataset_type: "VID" veya "DET".

        Returns:
            Bulunan veri seti dizini yolu, yoksa None.
        """
        pattern = os.path.join(datasets_dir, f"*VisDrone*{dataset_type}*")
        matches = glob(pattern)
        if matches:
            return matches[0]
        return None

    # =========================================================================
    #  VID YÜKLEME (Görev 2 — Sıralı Kareler)
    # =========================================================================

    def _load_video_sequence(self, vid_dir: str) -> None:
        """
        VID veri setinden rastgele bir sekans seçer ve sıralı kareleri listeler.

        Args:
            vid_dir: VisDrone VID veri seti kök dizini.
        """
        sequences_dir = os.path.join(vid_dir, "sequences")
        if not os.path.isdir(sequences_dir):
            self.log.error(f"'sequences' klasörü bulunamadı: {sequences_dir}")
            return

        # Mevcut sekansları listele
        sequence_dirs = [
            d for d in os.listdir(sequences_dir)
            if os.path.isdir(os.path.join(sequences_dir, d))
        ]

        if not sequence_dirs:
            self.log.error("Hiçbir video sekansı bulunamadı!")
            return

        # Rastgele bir sekans seç
        chosen = random.choice(sequence_dirs)
        self._sequence_name = chosen
        seq_path = os.path.join(sequences_dir, chosen)

        self.log.info(f"Sekans seçildi: {chosen} ({len(sequence_dirs)} seçenek arasından)")

        # Kareleri sıralı listele
        frames = sorted([
            os.path.join(seq_path, f)
            for f in os.listdir(seq_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not frames:
            self.log.error(f"Sekansta görüntü bulunamadı: {seq_path}")
            return

        self._frames = frames
        self._mode = "vid"
        self.log.info(f"Sıralı kareler yüklendi: {len(frames)} adet")

    # =========================================================================
    #  DET YÜKLEME (Görev 1 — Tekil Fotoğraflar)
    # =========================================================================

    def _load_detection_images(self, det_dir: str) -> None:
        """
        DET veri setinden rastgele N fotoğraf seçer.

        Args:
            det_dir: VisDrone DET veri seti kök dizini.
        """
        images_dir = os.path.join(det_dir, "images")
        if not os.path.isdir(images_dir):
            self.log.error(f"'images' klasörü bulunamadı: {images_dir}")
            return

        # Tüm görüntüleri listele
        all_images = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not all_images:
            self.log.error(f"Görüntü bulunamadı: {images_dir}")
            return

        # Rastgele örnekle
        sample_size = min(Settings.SIMULATION_DET_SAMPLE_SIZE, len(all_images))
        selected = sorted(random.sample(all_images, sample_size))

        self._frames = selected
        self._mode = "det"
        self.log.info(
            f"Rastgele {sample_size} fotoğraf seçildi "
            f"(toplam {len(all_images)} içinden)"
        )

    # =========================================================================
    #  ITERATOR PROTOCOL
    # =========================================================================

    def __len__(self) -> int:
        """Toplam kare sayısını döndürür."""
        return len(self._frames)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterator'ı sıfırlar ve döndürür."""
        self._index = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """
        Sıradaki kareyi yükler ve döndürür.

        Returns:
            Dict:
                - "frame": numpy BGR görüntüsü
                - "frame_idx": kare numarası (0-based)
                - "filename": dosya adı
                - "mode": "vid" veya "det"
                - "gps_health": simüle GPS sağlığı (VID: alternating, DET: 1)

        Raises:
            StopIteration: Tüm kareler işlendiğinde.
        """
        # Bozuk görselleri atla — recursive __next__ yerine döngü
        # (çok sayıda bozuk dosya olursa RecursionError'dan kaçınır)
        while self._index < len(self._frames):
            frame_path = self._frames[self._index]
            frame = cv2.imread(frame_path)

            if frame is None:
                self.log.warn(f"Görüntü okunamadı, atlanıyor: {frame_path}")
                self._index += 1
                continue

            # GPS sağlığını simüle et — Şartname Bölüm 3.2.2:
            # İlk 1 dakika (450 kare @ 7.5fps) GPS kesinlikle sağlıklı.
            # Son 4 dakikada (1800 kare) sağlıksız duruma geçebilir.
            # DET modunda: her zaman GPS sağlıklı (tekil kareler)
            if self._mode == "vid":
                if self._index < 450:
                    # İlk 1 dakika — kesinlikle sağlıklı
                    gps_health = 1
                else:
                    # 450+ frame: sağlıksız geçişleri simüle et
                    # Deterministik pattern: 100 kare sağlıklı, 200 kare sağlıksız
                    cycle_pos = (self._index - 450) % 300
                    gps_health = 1 if cycle_pos < 100 else 0
            else:
                gps_health = 1

            result: Dict[str, Any] = {
                "frame": frame,
                "frame_idx": self._index,
                "filename": os.path.basename(frame_path),
                "mode": self._mode,
                "gps_health": gps_health,
                # Sunucu formatını taklit et — localization.py beklediği key isimleri
                # Şartname: GPS sağlıksız olduğunda translation = "NaN"
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
        """Aktif veri seti modunu döndürür: 'vid' veya 'det'."""
        return self._mode

    @property
    def is_ready(self) -> bool:
        """Veri seti başarıyla yüklenmiş mi?"""
        return len(self._frames) > 0
