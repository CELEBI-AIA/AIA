"""
TEKNOFEST Havacılıkta Yapay Zeka - Ana Orkestrasyon Dosyası
============================================================
"""

import argparse
import os
import signal
import sys
import time
import traceback
from collections import Counter
from typing import Dict, Optional

import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import Settings
from src.detection import ObjectDetector
from src.localization import VisualOdometry
from src.movement import MovementEstimator
from src.runtime_profile import apply_runtime_profile
from src.utils import Logger, Visualizer

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     🛩️  TEKNOFEST 2026 - HAVACILIKTA YAPAY ZEKA YARIŞMASI    ║
║     ──────────────────────────────────────────────────────    ║
║     Nesne Tespiti (Görev 1) + Konum Kestirimi (Görev 2)     ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_system_info(log: Logger, simulate: bool = False) -> None:
    print(BANNER)
    log.info(f"Working Directory : {PROJECT_ROOT}")
    log.info(f"Mode             : {'SIMULATION' if simulate else 'COMPETITION'}")
    log.info(f"Debug            : {'ON' if Settings.DEBUG else 'OFF'}")
    log.info(f"Model            : {Settings.MODEL_PATH}")
    log.info(f"Device           : {Settings.DEVICE}")
    log.info(f"FP16             : {'ON' if Settings.HALF_PRECISION else 'OFF'}")
    log.info(f"TTA              : {'ON' if Settings.AUGMENTED_INFERENCE else 'OFF'}")

    if torch.cuda.is_available():
        log.success(f"GPU              : {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.success(f"GPU Memory       : {mem_total:.1f} GB")
    else:
        log.warn("GPU              : NOT FOUND, running on CPU")


class FPSCounter:
    def __init__(self, report_interval: int = 10) -> None:
        self.report_interval = report_interval
        self.frame_count: int = 0
        self.start_time: float = time.time()
        self.log = Logger("FPS")

    def tick(self) -> Optional[float]:
        self.frame_count += 1
        if self.frame_count % self.report_interval == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.log.info(
                f"Frame: {self.frame_count} | FPS: {fps:.2f} | Elapsed: {elapsed:.1f}s"
            )
            return fps
        return None


def run_simulation(
    log: Logger,
    prefer_vid: bool = True,
    show: bool = False,
    save: bool = False,
) -> None:
    from src.data_loader import DatasetLoader

    log.info("Initializing modules...")

    try:
        loader = DatasetLoader(prefer_vid=prefer_vid)
        if not loader.is_ready:
            log.error("Dataset loading failed, exiting.")
            return

        detector = ObjectDetector()
        odometry = VisualOdometry()
        movement = MovementEstimator()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        visualizer = Visualizer()
        if save:
            os.makedirs(Settings.DEBUG_OUTPUT_DIR, exist_ok=True)
            log.info(f"Saving frames to: {Settings.DEBUG_OUTPUT_DIR}")

        log.success("All modules initialized ✓")

    except Exception as exc:
        log.error(f"Initialization error: {exc}")
        log.error(f"Stack trace:\n{traceback.format_exc()}")
        return

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        log.warn("Shutdown signal received, stopping loop...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for frame_info in loader:
            if not running:
                break

            if fps_counter.frame_count >= Settings.MAX_FRAMES:
                log.success(f"Max frame limit reached ({Settings.MAX_FRAMES})")
                break

            try:
                frame = frame_info["frame"]
                frame_idx = frame_info["frame_idx"]
                server_data = frame_info["server_data"]
                gps_health = frame_info["gps_health"]

                detected_objects = detector.detect(frame)
                detected_objects = movement.annotate(detected_objects)
                position = odometry.update(frame, server_data)

                _print_simulation_result(log, frame_idx, detected_objects, position, gps_health)

                if show or save:
                    annotated = visualizer.draw_detections(
                        frame,
                        detected_objects,
                        frame_id=str(frame_idx),
                        position=position,
                        save_to_disk=not save,
                    )

                    mode_text = "GPS" if gps_health == 1 else "Optical Flow"
                    cv2.putText(
                        annotated,
                        f"Mode: {mode_text}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if gps_health else (0, 165, 255),
                        2,
                    )

                    if show:
                        cv2.imshow("TEKNOFEST - Simulation", annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            log.info("Window closed by user (q/ESC)")
                            break

                    if save:
                        save_path = os.path.join(
                            Settings.DEBUG_OUTPUT_DIR,
                            f"frame_{frame_idx:04d}.jpg",
                        )
                        cv2.imwrite(save_path, annotated)

                fps_counter.tick()

            except Exception as exc:
                log.error(f"Frame {frame_info.get('frame_idx', '?')} error: {exc}")
                log.error(f"Stack trace:\n{traceback.format_exc()}")
                continue

    finally:
        log.info("Cleaning resources...")
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        if save:
            log.success(f"Frames saved: {Settings.DEBUG_OUTPUT_DIR}/")

        _print_summary(log, fps_counter)


def _print_simulation_result(
    log: Logger,
    frame_idx: int,
    detected_objects: list,
    position: dict,
    gps_health: int,
) -> None:
    cls_counts = Counter(obj["cls"] for obj in detected_objects)
    tasit = cls_counts.get("0", 0)
    insan = cls_counts.get("1", 0)
    uap = cls_counts.get("2", 0)
    uai = cls_counts.get("3", 0)

    loc_mode = "GPS" if gps_health == 1 else "OF"
    pos_str = (
        f"x={position['x']:+.1f}m "
        f"y={position['y']:+.1f}m "
        f"z={position['z']:.0f}m"
    )

    log.success(
        f"Frame: {frame_idx:04d} | "
        f"Det: {len(detected_objects)} "
        f"({tasit} Vehicle, {insan} Human"
        f"{f', {uap} UAP' if uap else ''}"
        f"{f', {uai} UAI' if uai else ''}) | "
        f"Pos: {pos_str} ({loc_mode})"
    )


def run_competition(log: Logger) -> None:
    from src.network import FrameFetchStatus, NetworkManager

    log.info("Initializing modules...")

    try:
        network = NetworkManager(simulation_mode=False)
        detector = ObjectDetector()
        odometry = VisualOdometry()
        movement = MovementEstimator()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        visualizer: Optional[Visualizer] = Visualizer() if Settings.DEBUG else None

        log.success("All modules initialized ✓")

    except Exception as exc:
        log.error(f"Initialization error: {exc}")
        log.error("System startup failed, exiting.")
        return

    if not network.start_session():
        log.error("Server session start failed")
        return

    running = True
    transient_failures = 0
    transient_budget = max(10, Settings.MAX_RETRIES * 5)

    kpi_counters: Dict[str, int] = {
        "send_ok": 0,
        "send_fail": 0,
        "mode_gps": 0,
        "mode_of": 0,
    }

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        log.warn("Shutdown signal received (Ctrl+C)")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running:
            try:
                if fps_counter.frame_count >= Settings.MAX_FRAMES:
                    log.success(
                        f"Max frame count reached ({Settings.MAX_FRAMES}), session complete"
                    )
                    break

                fetch_result = network.get_frame()

                if fetch_result.status == FrameFetchStatus.END_OF_STREAM:
                    log.info("End of stream confirmed by server (204)")
                    break

                if fetch_result.status == FrameFetchStatus.TRANSIENT_ERROR:
                    transient_failures += 1
                    delay = min(5.0, Settings.RETRY_DELAY * (2 ** min(transient_failures, 4)))
                    log.warn(
                        f"Transient frame fetch failure {transient_failures}/{transient_budget}; "
                        f"retrying in {delay:.1f}s"
                    )
                    if transient_failures >= transient_budget:
                        log.error("Transient failure budget exceeded, aborting session")
                        break
                    time.sleep(delay)
                    continue

                if fetch_result.status == FrameFetchStatus.FATAL_ERROR:
                    log.error(
                        f"Fatal frame fetch error: {fetch_result.error_type} "
                        f"(http={fetch_result.http_status})"
                    )
                    break

                frame_data = fetch_result.frame_data or {}
                transient_failures = 0

                frame_id = frame_data.get("frame_id", "unknown")
                frame = network.download_image(frame_data)
                if frame is None:
                    log.warn(f"Frame {frame_id}: image download failed, skipping")
                    continue

                detected_objects = detector.detect(frame)
                detected_objects = movement.annotate(detected_objects)

                position = odometry.update(frame, frame_data)
                detected_translation = {
                    "translation_x": position["x"],
                    "translation_y": position["y"],
                    "translation_z": position["z"],
                }

                success = network.send_result(
                    frame_id,
                    detected_objects,
                    detected_translation,
                    frame_shape=frame.shape,
                )

                if success:
                    kpi_counters["send_ok"] += 1
                else:
                    kpi_counters["send_fail"] += 1
                    log.warn(f"Frame {frame_id}: result send failed")

                try:
                    gps_health = int(float(frame_data.get("gps_health", 0)))
                except (TypeError, ValueError):
                    gps_health = 0

                if gps_health == 1:
                    kpi_counters["mode_gps"] += 1
                else:
                    kpi_counters["mode_of"] += 1

                if Settings.DEBUG and visualizer is not None:
                    visualizer.draw_detections(
                        frame,
                        detected_objects,
                        frame_id=str(frame_id),
                        position=position,
                    )

                fps_counter.tick()

                interval = max(1, int(Settings.COMPETITION_RESULT_LOG_INTERVAL))
                if fps_counter.frame_count % interval == 0:
                    _print_competition_result(
                        log=log,
                        frame_id=frame_id,
                        detected_objects=detected_objects,
                        success=success,
                        frame_data=frame_data,
                        position=position,
                    )

                if Settings.LOOP_DELAY > 0:
                    time.sleep(Settings.LOOP_DELAY)

            except KeyboardInterrupt:
                log.warn("Interrupted by user")
                break
            except Exception as exc:
                log.error(f"Runtime error: {exc}")
                log.error(f"Stack trace:\n{traceback.format_exc()}")
                time.sleep(0.5)

    finally:
        log.info("Cleaning resources...")
        if Settings.DEBUG and visualizer is not None:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        _print_summary(log, fps_counter, kpi_counters=kpi_counters)


def _print_competition_result(
    log: Logger,
    frame_id,
    detected_objects: list,
    success: bool,
    frame_data: dict,
    position: dict,
) -> None:
    try:
        gps_health = int(float(frame_data.get("gps_health", 0)))
    except (TypeError, ValueError):
        gps_health = 0
    mode = "GPS" if gps_health == 1 else "OF"

    def _safe_float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    x = _safe_float(position.get("x", 0.0))
    y = _safe_float(position.get("y", 0.0))
    z = _safe_float(position.get("z", 0.0))

    send_status = "OK" if success else "FAIL"
    log.info(
        f"Frame: {frame_id} | Obj: {len(detected_objects)} | "
        f"Send: {send_status} | Mode: {mode} | "
        f"Pos: x={x:+.1f} y={y:+.1f} z={z:.1f}"
    )


def _print_summary(
    log: Logger,
    fps_counter: FPSCounter,
    kpi_counters: Optional[Dict[str, int]] = None,
) -> None:
    log.info("-" * 50)
    log.info(f"Total processed frames: {fps_counter.frame_count}")
    elapsed = time.time() - fps_counter.start_time
    if elapsed > 0:
        avg_fps = fps_counter.frame_count / elapsed
        log.info(f"Average FPS: {avg_fps:.2f}")
    log.info(f"Total elapsed: {elapsed:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("GPU cache cleared")

    if kpi_counters is not None:
        log.info(
            "KPI: "
            f"Send OK={kpi_counters.get('send_ok', 0)} | "
            f"Send FAIL={kpi_counters.get('send_fail', 0)} | "
            f"Mode GPS={kpi_counters.get('mode_gps', 0)} | "
            f"Mode OF={kpi_counters.get('mode_of', 0)}"
        )

    log.success("System shutdown complete")


def _ask_choice(prompt: str, options: dict) -> str:
    print()
    print(prompt)
    for key, desc in options.items():
        print(f"  [{key}] {desc}")
    print()

    while True:
        choice = input("  Selection: ").strip()
        if choice in options:
            return choice
        print(f"  Invalid selection, choose one of: {', '.join(options.keys())}")


def show_interactive_menu() -> dict:
    print("\n" + "=" * 56)
    print("  RUN MODE SELECTION")
    print("=" * 56)

    mode = _ask_choice(
        "  Choose run mode:",
        {
            "1": "Competition (server)",
            "2": "Simulation VID (sequential frames)",
            "3": "Simulation DET (single images)",
        },
    )

    if mode == "1":
        return {"mode": "competition", "prefer_vid": True, "show": False, "save": False}

    prefer_vid = mode == "2"

    output = _ask_choice(
        "  How do you want outputs?",
        {
            "1": "Terminal only",
            "2": "Show window",
            "3": "Save images",
            "4": "Show window + Save images",
        },
    )

    show = output in ("2", "4")
    save = output in ("3", "4")

    return {"mode": "simulate", "prefer_vid": prefer_vid, "show": show, "save": save}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEKNOFEST AIA Runtime")
    parser.add_argument(
        "--mode",
        choices=["competition", "simulate_vid", "simulate_det"],
        default="competition",
        help="Run mode (default: competition)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive menu flow",
    )
    parser.add_argument(
        "--deterministic-profile",
        choices=["off", "balanced", "max"],
        default="balanced",
        help="Runtime determinism profile",
    )
    parser.add_argument("--show", action="store_true", help="Show simulation window")
    parser.add_argument("--save", action="store_true", help="Save simulation images")
    return parser.parse_args()


def main() -> None:
    log = Logger("Main")
    args = parse_args()

    apply_runtime_profile(args.deterministic_profile)

    print(BANNER)

    if args.interactive:
        choices = show_interactive_menu()
    else:
        if args.mode == "competition":
            choices = {
                "mode": "competition",
                "prefer_vid": True,
                "show": False,
                "save": False,
            }
        elif args.mode == "simulate_vid":
            choices = {
                "mode": "simulate",
                "prefer_vid": True,
                "show": args.show,
                "save": args.save,
            }
        else:
            choices = {
                "mode": "simulate",
                "prefer_vid": False,
                "show": args.show,
                "save": args.save,
            }

    simulate = choices["mode"] == "simulate"
    print_system_info(log, simulate=simulate)

    if simulate:
        run_simulation(
            log,
            prefer_vid=choices["prefer_vid"],
            show=choices["show"],
            save=choices["save"],
        )
    else:
        run_competition(log)


if __name__ == "__main__":
    main()
