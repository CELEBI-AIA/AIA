"""TEKNOFEST Havacılıkta Yapay Zeka — Ana orkestrasyon.
Simülasyon: datasets/ içinden kare yükler. Yarışma: sunucudan frame alır, sonuç gönderir."""

import argparse
import os
import signal
import sys
import time
from collections import Counter
from typing import Dict, Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import Settings
from src.detection import ObjectDetector
from src.localization import VisualOdometry
from src.movement import MovementEstimator
from src.resilience import SessionResilienceController
from src.runtime_profile import apply_runtime_profile
from src.send_state import apply_send_result_status
from src.utils import Logger, Visualizer
from src.frame_context import FrameContext
from typing import Any

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     🛩️  TEKNOFEST 2026 - HAVACILIKTA YAPAY ZEKA YARIŞMASI    ║
║     ──────────────────────────────────────────────────────    ║
║     Nesne Tespiti (Görev 1) + Konum Kestirimi (Görev 2)     ║
║     Görüntü Eşleme (Görev 3)                                ║
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
    seed: Optional[int] = None,
    sequence: Optional[str] = None,
) -> None:
    from src.data_loader import DatasetLoader

    log.info("Initializing modules...")

    try:
        loader = DatasetLoader(prefer_vid=prefer_vid, seed=seed, sequence=sequence)
        if not loader.is_ready:
            log.error("Dataset loading failed, exiting.")
            return

        detector = ObjectDetector()
        odometry = VisualOdometry()
        movement = MovementEstimator()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        image_matcher = None
        if Settings.TASK3_ENABLED:
            from src.image_matcher import ImageMatcher
            image_matcher = ImageMatcher()
            loaded = image_matcher.load_references_from_directory()
            if loaded == 0:
                log.warn("Görev 3: Referans obje bulunamadı, Simülasyonda Görev 3 pasif")
                image_matcher = None

        visualizer = Visualizer()
        if save:
            os.makedirs(Settings.DEBUG_OUTPUT_DIR, exist_ok=True)
            log.info(f"Saving frames to: {Settings.DEBUG_OUTPUT_DIR}")

        log.success("All modules initialized successfully")

    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        log.error(f"Initialization error: {exc}")
        return

    running = True

    def signal_handler(sig, frame) -> None:
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
                should_stop = _process_simulation_step(
                    log,
                    frame_info,
                    detector,
                    movement,
                    odometry,
                    image_matcher,
                    visualizer,
                    show,
                    save,
                )
                if should_stop:
                    break
                fps_counter.tick()

            except Exception as exc:
                log.error(f"Frame {frame_info.get('frame_idx', '?')} error: {exc}")
                continue

    finally:
        log.info("Cleaning resources...")
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        if save:
            log.success(f"Frames saved: {Settings.DEBUG_OUTPUT_DIR}/")

        _print_summary(log, fps_counter)


def _process_simulation_step(
    log: Logger,
    frame_info: dict,
    detector: ObjectDetector,
    movement: MovementEstimator,
    odometry: VisualOdometry,
    image_matcher: Any,
    visualizer: Any,
    show: bool,
    save: bool,
) -> bool:
    frame = frame_info["frame"]
    frame_idx = frame_info["frame_idx"]
    server_data = frame_info["server_data"]
    gps_health = frame_info["gps_health"]

    frame_ctx = FrameContext(frame)
    detected_objects = detector.detect(frame)
    detected_objects = movement.annotate(detected_objects, frame_ctx=frame_ctx)
    position = odometry.update(frame_ctx, server_data)

    if image_matcher is not None:
        _ = image_matcher.match(frame)

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
            try:
                cv2.imshow("TEKNOFEST - Simulation", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    log.info("Window closed by user (q/ESC)")
                    return True
            except cv2.error:
                pass

        if save:
            save_path = os.path.join(
                Settings.DEBUG_OUTPUT_DIR,
                f"frame_{frame_idx:04d}.jpg",
            )
            cv2.imwrite(save_path, annotated)

    return False


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
    from src.network import NetworkManager

    log.info("Initializing modules...")

    try:
        network = NetworkManager(simulation_mode=False)
        detector = ObjectDetector()
        odometry = VisualOdometry()
        movement = MovementEstimator()
        fps_counter = FPSCounter(report_interval=Settings.FPS_REPORT_INTERVAL)

        image_matcher = None
        if Settings.TASK3_ENABLED:
            from src.image_matcher import ImageMatcher
            image_matcher = ImageMatcher()
            loaded = image_matcher.load_references_from_directory()
            if loaded == 0:
                log.warn("Görev 3: Referans obje bulunamadı, detected_undefined_objects boş gönderilecek")

        visualizer: Optional[Visualizer] = Visualizer() if Settings.DEBUG else None

        log.success("All modules initialized successfully")

    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        log.error(f"Initialization error: {exc}")
        log.error("System startup failed, exiting.")
        return

    if not network.start_session():
        log.error("Server session start failed")
        return

    server_refs = network.get_task3_references()
    if image_matcher is not None and server_refs:
        ref_list = []
        for r in server_refs:
            if not isinstance(r, dict):
                continue
            obj_id = int(r.get("object_id", len(ref_list) + 1))
            if "path" in r and r["path"]:
                p = r["path"]
                if not os.path.isabs(p):
                    p = os.path.join(PROJECT_ROOT, p)
                ref_list.append({"object_id": obj_id, "path": p, "label": f"ref_{obj_id}"})
            elif "image_base64" in r and r["image_base64"]:
                try:
                    import base64
                    arr = np.frombuffer(base64.b64decode(r["image_base64"]), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        ref_list.append({"object_id": obj_id, "image": img, "label": f"ref_{obj_id}"})
                except Exception:
                    pass
        if ref_list:
            loaded = image_matcher.load_references(ref_list)
            log.info(f"Görev 3: Sunucudan {loaded} referans obje yüklendi")

    running = True
    transient_failures = 0
    transient_budget = max(10, Settings.MAX_RETRIES * 5)
    ack_failures = 0
    ack_failure_budget = max(20, Settings.MAX_RETRIES * 10)
    consecutive_permanent_rejects = 0
    PERMANENT_REJECT_ABORT_THRESHOLD = 5
    consecutive_duplicate_frames = 0
    CONSECUTIVE_DUPLICATE_ABORT_THRESHOLD = 5
    resilience = SessionResilienceController(log=log)

    kpi_counters: Dict[str, int] = {
        "send_ok": 0,
        "send_fail": 0,
        "send_fallback_ok": 0,
        "send_permanent_reject": 0,
        "payload_preflight_reject_count": 0,
        "payload_clipped_count": 0,
        "mode_gps": 0,
        "mode_of": 0,
        "degrade_frames": 0,
        "frame_duplicate_drop": 0,
        "timeout_fetch": 0,
        "timeout_image": 0,
        "timeout_submit": 0,
        "consecutive_duplicate_abort": 0,
    }
    pending_result: Optional[Dict] = None

    def signal_handler(sig, frame) -> None:
        nonlocal running
        running = False
        log.warn("Shutdown signal received (Ctrl+C)")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    import concurrent.futures

    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        fetch_future = None
        submit_future = None
        
        while running:
            try:
                abort_reason = resilience.should_abort()
                if abort_reason:
                    log.error(f"Resilience abort: {abort_reason}")
                    break

                if fps_counter.frame_count >= Settings.MAX_FRAMES:
                    log.success(
                        f"Max frame count reached ({Settings.MAX_FRAMES}), session complete"
                    )
                    break

                # Bekleyen sonuç varsa önce gönder (fetch ile çakışmayı önle)
                if pending_result is not None and submit_future is None:
                    submit_future = executor.submit(
                        _submit_competition_step,
                        log, network, resilience, kpi_counters, pending_result,
                        ack_failures, ack_failure_budget,
                        consecutive_permanent_rejects, PERMANENT_REJECT_ABORT_THRESHOLD,
                    )

                if submit_future is not None and submit_future.done():
                    result = submit_future.result()
                    pending_result = result[0]
                    ack_failures = result[1]
                    action_result = result[2]
                    consecutive_permanent_rejects = result[3]
                    submit_future = None
                    if action_result == "break":
                        break
                    elif action_result == "continue":
                        continue
                    
                    if not (isinstance(action_result, tuple) and len(action_result) == 2):
                        log.error(f"Unexpected action_result type: {type(action_result)}, skipping frame")
                        continue
                    _, success_info = action_result
                    
                    try:
                        gps_health = int(float(success_info["frame_data"].get("gps_health", 0)))
                    except (TypeError, ValueError):
                        gps_health = 0

                    if gps_health == 1:
                        kpi_counters["mode_gps"] += 1
                    else:
                        kpi_counters["mode_of"] += 1

                    if Settings.DEBUG and visualizer is not None and success_info["frame_for_debug"] is not None:
                        visualizer.draw_detections(
                            success_info["frame_for_debug"],
                            success_info["detected_objects"],
                            frame_id=str(success_info["frame_id"]),
                            position=success_info["position"],
                        )

                    fps_counter.tick()

                    interval = max(1, int(Settings.COMPETITION_RESULT_LOG_INTERVAL))
                    if fps_counter.frame_count % interval == 0:
                        _print_competition_result(
                            log=log,
                            frame_id=success_info["frame_id"],
                            detected_objects=success_info["detected_objects"],
                            send_status="SUCCESS",
                            position=success_info["position"],
                            gps_health=gps_health,
                        )

                    if Settings.LOOP_DELAY > 0:
                        time.sleep(Settings.LOOP_DELAY)

                elif submit_future is not None and not submit_future.done():
                    time.sleep(0.01)
                    continue

                elif pending_result is None:
                    # Yeni frame al
                    if fetch_future is None:
                        fetch_future = executor.submit(
                            _fetch_competition_step,
                            log, network, detector, movement, odometry, image_matcher,
                            resilience, kpi_counters, transient_failures, transient_budget
                        )
                    
                    if fetch_future.done():
                        fetch_res, tf_new, action, is_dup = fetch_future.result()
                        fetch_future = None
                        transient_failures = tf_new
                        if action == "continue":
                            time.sleep(min(max(0.2, Settings.RETRY_DELAY), 5.0) if fetch_res is None else 0)
                            continue
                        elif action == "break":
                            break
                        if is_dup:
                            consecutive_duplicate_frames += 1
                            if consecutive_duplicate_frames >= CONSECUTIVE_DUPLICATE_ABORT_THRESHOLD:
                                kpi_counters["consecutive_duplicate_abort"] = consecutive_duplicate_frames
                                log.error(f"Ardışık {consecutive_duplicate_frames} duplicate frame, oturum sonlandırılıyor")
                                break
                        else:
                            consecutive_duplicate_frames = 0
                        pending_result = fetch_res
                    else:
                        time.sleep(0.01)
                        continue

            except KeyboardInterrupt:
                log.warn("Interrupted by user")
                break
            except Exception as exc:
                log.error(f"Runtime error: {exc}")
                if "pytest" in sys.modules and getattr(sys, "last_type", None) is None:
                    raise
                time.sleep(0.5)

    finally:
        resilience_stats = resilience.finalize()
        log.info("Cleaning resources...")
        if Settings.DEBUG and visualizer is not None:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        _print_summary(
            log,
            fps_counter,
            kpi_counters=kpi_counters,
            resilience_stats={
                "breaker_open_count": resilience_stats.breaker_open_count,
                "degrade_entries": resilience_stats.degrade_entries,
                "degrade_frames": resilience_stats.degrade_frames,
                "recovered_count": resilience_stats.recovered_count,
                "transient_wall_time_sec": resilience_stats.transient_wall_time_sec,
            },
        )


def _print_competition_result(
    log: Logger,
    frame_id,
    detected_objects: list,
    send_status: str,
    position: dict,
    gps_health: int,
) -> None:
    mode = "GPS" if gps_health == 1 else "OF"

    def _safe_float(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    x = _safe_float(position.get("x", 0.0))
    y = _safe_float(position.get("y", 0.0))
    z = _safe_float(position.get("z", 0.0))

    send_status_text = "OK" if send_status in {"acked", "fallback_acked", "SUCCESS"} else "FAIL"
    log.info(
        f"Frame: {frame_id} | Obj: {len(detected_objects)} | "
        f"Send: {send_status_text} ({send_status}) | Mode: {mode} | "
        f"Pos: x={x:+.1f} y={y:+.1f} z={z:.1f}"
    )


def _print_summary(
    log: Logger,
    fps_counter: FPSCounter,
    kpi_counters: Optional[Dict[str, int]] = None,
    resilience_stats: Optional[Dict[str, float]] = None,
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

    def val_str(val):
        return f"{val:.0f}" if isinstance(val, (float, int)) else str(val)

    if kpi_counters is not None:
        log.info(
            "KPI: "
            f"Send OK={kpi_counters.get('send_ok', 0)} | "
            f"Send FAIL={kpi_counters.get('send_fail', 0)} | "
            f"Fallback ACK={kpi_counters.get('send_fallback_ok', 0)} | "
            f"Permanent Reject={kpi_counters.get('send_permanent_reject', 0)} | "
            f"Preflight Reject={kpi_counters.get('payload_preflight_reject_count', 0)} | "
            f"Payload Clipped={kpi_counters.get('payload_clipped_count', 0)} | "
            f"Mode OF={kpi_counters.get('mode_of', 0)} | "
            f"Degrade Frames={kpi_counters.get('degrade_frames', 0)} | "
            f"DupDrop={kpi_counters.get('frame_duplicate_drop', 0)} | "
            f"Timeouts(fetch/image/submit)="
            f"{kpi_counters.get('timeout_fetch', 0)}/"
            f"{kpi_counters.get('timeout_image', 0)}/"
            f"{kpi_counters.get('timeout_submit', 0)}"
        )
        log.info(
            f"Payload Size   : Max {val_str(kpi_counters.get('max_payload_bytes'))} bytes "
            f"(Avg: {val_str(kpi_counters.get('avg_payload_bytes'))} bytes)"
        )

    if resilience_stats is not None:
        log.info(
            "Resilience: "
            f"Breaker Open Count={int(resilience_stats.get('breaker_open_count', 0))} | "
            f"Degrade Entries={int(resilience_stats.get('degrade_entries', 0))} | "
            f"Recovery Count={int(resilience_stats.get('recovered_count', 0))} | "
            f"Transient Wall Time={float(resilience_stats.get('transient_wall_time_sec', 0.0)):.1f}s"
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
    parser.add_argument("--seed", type=int, default=None, help="Deterministik simülasyon için rastgele seed")
    parser.add_argument("--sequence", type=str, default=None, help="VID modunda seçilecek sekans adı (örn. uav0000123)")
    return parser.parse_args()


def main() -> None:
    log = Logger("Main")
    args = parse_args()

    requested_profile = args.deterministic_profile
    effective_profile = requested_profile
    if args.mode == "competition" and requested_profile != "max":
        log.warn(
            "Competition mode requires deterministic-profile=max; "
            f"overriding requested profile '{requested_profile}' -> 'max'"
        )
        effective_profile = "max"

    apply_runtime_profile(effective_profile, requested_profile=requested_profile)

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
            seed=args.seed,
            sequence=args.sequence,
        )
    else:
        run_competition(log)


def _fetch_competition_step(
    log: Logger, network: Any, detector: Any, movement: Any, odometry: Any, image_matcher: Any,
    resilience: Any, kpi_counters: dict, transient_failures: int, transient_budget: int
):
    from src.network import FrameFetchStatus
    import time

    if not resilience.before_fetch():
        cooldown_left = resilience.open_cooldown_remaining()
        wait_s = min(max(0.2, Settings.RETRY_DELAY), max(0.2, cooldown_left))
        log.warn(f"Circuit breaker OPEN; waiting cooldown ({cooldown_left:.1f}s remaining, sleep={wait_s:.1f}s)")
        return None, transient_failures, "continue", False

    fetch_result = network.get_frame()
    timeout_snapshot = network.consume_timeout_counters()
    kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
    kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
    kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)

    if fetch_result.status == FrameFetchStatus.END_OF_STREAM:
        log.info("End of stream confirmed by server (204)")
        return None, transient_failures, "break", False

    if fetch_result.status == FrameFetchStatus.TRANSIENT_ERROR:
        transient_failures += 1
        resilience.on_fetch_transient()
        delay = min(5.0, Settings.RETRY_DELAY * (2 ** min(transient_failures, 4)))
        log.warn(f"Transient frame fetch failure {transient_failures}/{transient_budget}; retrying in {delay:.1f}s")
        if transient_failures >= transient_budget:
            log.warn("Transient failure budget reached; session stays alive under wall-clock circuit breaker policy")
        time.sleep(delay)
        return None, transient_failures, "continue", False

    if fetch_result.status == FrameFetchStatus.FATAL_ERROR:
        log.error(f"Fatal frame fetch error: {fetch_result.error_type} (http={fetch_result.http_status})")
        return None, transient_failures, "break", False

    frame_data = fetch_result.frame_data or {}
    transient_failures = 0
    degrade_mode = Settings.DEGRADE_FETCH_ONLY_ENABLED and resilience.is_degraded()
    frame_id = frame_data.get("frame_id", "unknown")

    if fetch_result.is_duplicate:
        kpi_counters["frame_duplicate_drop"] += 1
        log.warn(f"Frame {frame_id}: duplicate metadata detected. Processing normally per specification idempotency.")

    frame = None
    use_fallback = False

    if degrade_mode:
        kpi_counters["degrade_frames"] += 1
        degrade_seq = resilience.record_degraded_frame()
        heavy_every = max(1, int(Settings.DEGRADE_SEND_INTERVAL_FRAMES))
        should_try_heavy = (degrade_seq % heavy_every) == 0
        if should_try_heavy:
            frame = network.download_image(frame_data)
            timeout_snapshot = network.consume_timeout_counters()
            kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
            kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
            kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)
            if frame is None:
                use_fallback = True
                log.warn(f"Frame {frame_id}: degrade heavy pass image download failed, sending fallback result")
            else:
                log.info(f"Frame {frame_id}: degrade heavy pass (every {heavy_every} frames)")
        else:
            use_fallback = True
            log.info(f"Frame {frame_id}: degraded fetch-only fallback (slot {degrade_seq}/{heavy_every})")
    else:
        frame = network.download_image(frame_data)
        timeout_snapshot = network.consume_timeout_counters()
        kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
        kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
        kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)
        if frame is None:
            use_fallback = True
            log.warn(f"Frame {frame_id}: image download failed, sending fallback result")

    if use_fallback:
        gps_health = int(float(frame_data.get("gps_health", 0)))
        if gps_health == 0:
            last_position = odometry.get_last_of_position()
        else:
            last_position = odometry.get_position()
        pending_result = {
            "frame_id": frame_id, "frame_data": frame_data, "detected_objects": [],
            "frame": None, "position": last_position,
            "degraded": degrade_mode, "pending_ttl": 1 if degrade_mode else None,
            "detected_translation": {"translation_x": last_position["x"], "translation_y": last_position["y"], "translation_z": last_position["z"]},
            "frame_shape": None, "detected_undefined_objects": [],
            "is_duplicate": fetch_result.is_duplicate,
        }
    else:
        frame_ctx = FrameContext(frame)
        detected_objects = detector.detect(frame)
        detected_objects = movement.annotate(detected_objects, frame_ctx=frame_ctx)
        undefined_objects = []
        if image_matcher is not None:
            undefined_objects = image_matcher.match(frame)
        position = odometry.update(frame_ctx, frame_data)
        pending_result = {
            "frame_id": frame_id, "frame_data": frame_data, "detected_objects": detected_objects,
            "frame": frame, "position": position, "degraded": degrade_mode,
            "pending_ttl": 1 if degrade_mode else None,
            "detected_translation": {"translation_x": position["x"], "translation_y": position["y"], "translation_z": position["z"]},
            "frame_shape": frame.shape, "detected_undefined_objects": undefined_objects,
            "is_duplicate": fetch_result.is_duplicate,
        }

    return pending_result, transient_failures, "process", fetch_result.is_duplicate

def _submit_competition_step(
    log: Logger, network: Any, resilience: Any, kpi_counters: dict, pending_result: dict,
    ack_failures: int, ack_failure_budget: int,
    consecutive_permanent_rejects: int, permanent_reject_abort_threshold: int,
):
    import time
    frame_id = pending_result["frame_id"]
    frame_data = pending_result["frame_data"]
    detected_objects = pending_result["detected_objects"]
    
    send_status = network.send_result(
        frame_id, detected_objects, pending_result["detected_translation"],
        frame_data=frame_data, frame_shape=pending_result["frame_shape"],
        degrade=bool(pending_result.get("degraded", False)),
        detected_undefined_objects=pending_result.get("detected_undefined_objects"),
    )
    
    timeout_snapshot = network.consume_timeout_counters()
    kpi_counters["timeout_fetch"] += timeout_snapshot.get("fetch", 0)
    kpi_counters["timeout_image"] += timeout_snapshot.get("image", 0)
    kpi_counters["timeout_submit"] += timeout_snapshot.get("submit", 0)
    
    guard_snapshot = network.consume_payload_guard_counters()
    kpi_counters["payload_preflight_reject_count"] += guard_snapshot.get("preflight_reject", 0)
    kpi_counters["payload_clipped_count"] += guard_snapshot.get("payload_clipped", 0)

    pending_result_snapshot = dict(pending_result)

    pending_result, should_abort_session, success_cycle = apply_send_result_status(
        send_status=send_status, pending_result=pending_result, kpi_counters=kpi_counters,
    )

    if pending_result is None and not success_cycle:
        consecutive_permanent_rejects += 1
        log.warn(f"Frame {frame_id}: permanent rejected, frame dropped ({consecutive_permanent_rejects}/{permanent_reject_abort_threshold})")
        if consecutive_permanent_rejects >= permanent_reject_abort_threshold:
            log.error("Consecutive permanent reject threshold reached, aborting session")
            return None, ack_failures, "break", consecutive_permanent_rejects
        return None, ack_failures, "continue", consecutive_permanent_rejects

    if success_cycle:
        ack_failures = 0
        consecutive_permanent_rejects = 0
        resilience.on_success_cycle()
        success_info = {
            "frame_for_debug": pending_result_snapshot.get("frame"),
            "detected_objects": detected_objects,
            "position": pending_result_snapshot.get("position"),
            "frame_id": frame_id,
            "frame_data": frame_data
        }
        return pending_result, ack_failures, ("process", success_info), consecutive_permanent_rejects
    else:
        ack_failures += 1
        resilience.on_ack_failure()
        delay = min(5.0, Settings.RETRY_DELAY * (2 ** min(ack_failures, 4)))
        log.warn(f"Frame {frame_id}: result send failed ({send_status}), waiting ACK ({ack_failures}/{ack_failure_budget}); retrying in {delay:.1f}s")
        if ack_failures >= ack_failure_budget:
            log.warn("ACK failure budget reached; session stays alive under wall-clock circuit breaker policy")
        
        if pending_result is not None:
            pending_ttl = pending_result.get("pending_ttl")
            if pending_ttl is not None:
                pending_ttl = int(pending_ttl) - 1
                pending_result["pending_ttl"] = pending_ttl
                if pending_ttl <= 0:
                    log.warn(f"Frame {frame_id}: stale degraded pending result dropped after TTL")
                    pending_result = None
        
        time.sleep(delay)
        return pending_result, ack_failures, "continue", consecutive_permanent_rejects

if __name__ == "__main__":
    main()
