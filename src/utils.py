"""Logger (seviyeli log), Visualizer (bbox çizimi), log_json_to_disk (gelen/giden JSON kayıt)."""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any

import cv2
import numpy as np

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

from config.settings import Settings


class Logger:
    """Seviyeli log (DEBUG, INFO, WARN, ERROR, SUCCESS)."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def _timestamp(self) -> str:
        now = datetime.now()
        return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"

    def _print(self, level: str, color: str, message: str) -> None:
        ts = self._timestamp()
        prefix = f"[{ts}] [{level:^7}] [{self.module_name}]"
        if _HAS_COLORAMA:
            print(f"{color}{prefix}{Style.RESET_ALL} {message}")
        else:
            print(f"{prefix} {message}")

    def debug(self, message: str) -> None:
        if Settings.DEBUG:
            color = Fore.WHITE if _HAS_COLORAMA else ""
            self._print("DEBUG", color, message)

    def info(self, message: str) -> None:
        color = Fore.GREEN if _HAS_COLORAMA else ""
        self._print("INFO", color, message)

    def warn(self, message: str) -> None:
        color = Fore.YELLOW if _HAS_COLORAMA else ""
        self._print("WARN", color, message)

    def error(self, message: str) -> None:
        color = Fore.RED if _HAS_COLORAMA else ""
        self._print("ERROR", color, message)

    def success(self, message: str) -> None:
        color = Fore.CYAN if _HAS_COLORAMA else ""
        self._print("SUCCESS", color, message)


class Visualizer:
    """Debug: bbox, etiket, iniş durumu çizimi."""

    CLASS_COLORS: Dict[int, tuple] = {
        0: (0, 255, 0),      # Taşıt → Yeşil
        1: (255, 0, 0),      # İnsan → Mavi
        2: (255, 255, 0),    # UAP → Cyan
        3: (0, 0, 255),      # UAİ → Kırmızı
    }

    CLASS_NAMES: Dict[int, str] = {
        0: "Tasit",
        1: "Insan",
        2: "UAP",
        3: "UAI",
    }

    LANDING_LABELS: Dict[str, str] = {
        "-1": "",
        "0": " [UYGUN DEGIL]",
        "1": " [UYGUN]",
    }

    def __init__(self) -> None:
        os.makedirs(Settings.DEBUG_OUTPUT_DIR, exist_ok=True)
        self.log = Logger("Visualizer")
        self._save_counter: int = 0

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        frame_id: str = "unknown",
        position: Optional[Dict] = None,
        save_to_disk: bool = True,
    ) -> np.ndarray:
        annotated = frame.copy()

        for det in detections:
            cls_id = int(det.get("cls", -1))
            landing = det.get("landing_status", "-1")
            x1 = int(float(det.get("top_left_x", 0)))
            y1 = int(float(det.get("top_left_y", 0)))
            x2 = int(float(det.get("bottom_right_x", 0)))
            y2 = int(float(det.get("bottom_right_y", 0)))
            conf = det.get("confidence", 0.0)

            color = self.CLASS_COLORS.get(cls_id, (200, 200, 200))
            label_name = self.CLASS_NAMES.get(cls_id, "?")
            landing_txt = self.LANDING_LABELS.get(landing, "")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{label_name} {conf:.2f}{landing_txt}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

        if position:
            pos_text = (
                f"X:{position.get('x', 0):.2f}m "
                f"Y:{position.get('y', 0):.2f}m "
                f"Z:{position.get('z', 0):.2f}m"
            )
            cv2.putText(
                annotated, pos_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        if save_to_disk:
            self._save_counter += 1
            if self._save_counter % Settings.DEBUG_SAVE_INTERVAL == 0:
                save_path = os.path.join(Settings.DEBUG_OUTPUT_DIR, f"{frame_id}.jpg")
                cv2.imwrite(save_path, annotated)
                self.log.debug(f"Debug görsel kaydedildi: {save_path}")

        return annotated


def log_json_to_disk(
    data: Any,
    direction: str = "outgoing",
    tag: str = "general",
) -> None:
    try:
        os.makedirs(Settings.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_direction = _sanitize_log_component(direction)
        safe_tag = _sanitize_log_component(tag)
        filename = f"{timestamp}_{safe_direction}_{safe_tag}.json"
        filepath = os.path.join(Settings.LOG_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        _prune_old_logs(Settings.LOG_DIR)

    except Exception as exc:
        Logger("Logger").warn(f"JSON log write failed: {exc}")


def _sanitize_log_component(value: Any) -> str:
    text = str(value)
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    return sanitized[:80] if sanitized else "general"


def _prune_old_logs(log_dir: str) -> None:
    max_files = max(1, int(Settings.LOG_MAX_FILES))
    try:
        files = [
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.lower().endswith(".json")
        ]
    except Exception:
        return

    if len(files) <= max_files:
        return

    files.sort(key=lambda path: os.path.getmtime(path))
    for old_file in files[: len(files) - max_files]:
        try:
            os.remove(old_file)
        except Exception:
            continue
