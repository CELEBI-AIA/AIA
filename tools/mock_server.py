"""Yerel mock sunucu — yarışma formatında test."""

import json
import os
import sys
import threading
import time
from glob import glob
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def discover_frames() -> List[str]:
    datasets_dir = PROJECT_ROOT / "datasets"
    frames = []

    if datasets_dir.is_dir():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            found = sorted(glob(str(datasets_dir / "**" / ext), recursive=True))
            frames.extend(found)

    if not frames:
        bus_path = PROJECT_ROOT / "bus.jpg"
        if bus_path.is_file():
            frames = [str(bus_path)] * 100  # 100 kare simüle et

    return frames[:2250]


class MockServerHandler(BaseHTTPRequestHandler):
    frames = discover_frames()
    current_index = 0
    results_received = 0
    session_start = time.time()
    _lock = threading.Lock()

    def log_message(self, format: str, *args) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [MockServer] {format % args}")

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "":
            task3_refs = []
            ref_dir = PROJECT_ROOT / "datasets" / "task3_references"
            if ref_dir.is_dir():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for p in sorted(ref_dir.glob(ext))[:10]:
                        task3_refs.append({"object_id": len(task3_refs) + 1, "path": str(p)})
            payload = {
                "status": "ok",
                "server": "TEKNOFEST Mock Server",
                "total_frames": len(self.frames),
            }
            if task3_refs:
                payload["task3_references"] = task3_refs
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
            return

        if self.path.startswith("/next_frame"):
            self._handle_next_frame()
            return

        if self.path.startswith("/images/"):
            self._serve_image()
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path.startswith("/submit_result"):
            self._handle_submit_result()
            return

        self.send_error(404, "Not Found")

    def _handle_next_frame(self) -> None:
        cls = MockServerHandler
        with cls._lock:
            if cls.current_index >= len(cls.frames):
                self.send_response(204)
                self.end_headers()
                self.log_message("End of stream (204) — tüm kareler tamamlandı")
                return
            frame_path = cls.frames[cls.current_index]
            frame_id = cls.current_index

        if frame_id < 450:
            gps_health = 1
            tx = float(frame_id * 0.5)
            ty = float(frame_id * 0.1)
            tz = 50.0
        else:
            cycle = (frame_id - 450) % 300
            gps_health = 1 if cycle < 100 else 0
            if gps_health == 1:
                tx = float(frame_id * 0.5)
                ty = float(frame_id * 0.1)
                tz = 50.0
            else:
                tx = "NaN"
                ty = "NaN"
                tz = "NaN"

        rel_path = os.path.relpath(frame_path, PROJECT_ROOT).replace("\\", "/")

        frame_data = {
            "url": f"http://localhost:5000/frames/{frame_id}/",
            "image_url": f"/images/{rel_path}",
            "video_name": "mock_video_01",
            "session": "http://localhost:5000/session/1/",
            "frame_id": frame_id,
            "id": frame_id,
            "frame_url": f"http://localhost:5000/images/{rel_path}",
            "translation_x": tx,
            "translation_y": ty,
            "translation_z": tz,
            "gps_health_status": gps_health,
            "gps_health": gps_health,
        }

        with cls._lock:
            cls.current_index += 1

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(frame_data).encode())

    def _handle_submit_result(self) -> None:
        cls = MockServerHandler
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            result = json.loads(body)
            with cls._lock:
                cls.results_received += 1

            obj_count = len(result.get("detected_objects", []))
            undef_count = len(result.get("detected_undefined_objects", []))
            frame = result.get("frame", "?")

            self.log_message(
                f"Result #{cls.results_received}: frame={frame} | "
                f"objects={obj_count} | undefined={undef_count}"
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")

    def _serve_image(self) -> None:
        rel_path = self.path[len("/images/"):]
        
        try:
            abs_path = (PROJECT_ROOT / rel_path).resolve()
        except Exception:
            self.send_error(400, "Bad Request")
            return

        if not str(abs_path).startswith(str(PROJECT_ROOT.resolve())):
            self.send_error(403, "Forbidden")
            return

        if not abs_path.is_file():
            self.send_error(404, f"Image not found: {rel_path}")
            return

        ext = abs_path.suffix.lower()
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")

        with open(abs_path, "rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    host = "0.0.0.0"
    port = 5000

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     🧪 TEKNOFEST Mock Server — Yerel Test Sunucusu          ║
║     ──────────────────────────────────────────────────────   ║
║     http://{host}:{port}                                       ║
╚══════════════════════════════════════════════════════════════╝
""")

    frames = MockServerHandler.frames
    print(f"  📂 Toplam kare: {len(frames)}")
    if frames:
        print(f"  📁 İlk kare: {frames[0]}")
    else:
        print("  ⚠️  Hiçbir görüntü bulunamadı! datasets/ klasörüne veri koyun.")
        return

    print(f"\n  Dinleniyor: http://localhost:{port}")
    print("  Durdurmak için Ctrl+C\n")

    server = HTTPServer((host, port), MockServerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        elapsed = time.time() - MockServerHandler.session_start
        print("\n\n  Sunucu kapatıldı.")
        print(f"  Toplam süre: {elapsed:.1f}s")
        print(f"  Gönderilen kare: {MockServerHandler.current_index}")
        print(f"  Alınan sonuç: {MockServerHandler.results_received}")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
