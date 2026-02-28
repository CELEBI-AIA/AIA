# AIA SYSTEM AUDIT REPORT
**Generated:** Analysis Only Mode
**Target:** TEKNOFEST Aviation AI System (AIA)

---

## 1. Detected Issues

### Issue 1.1: Incorrect GPS Simulation Cycle
- **Description:** `data_loader.py` simulates GPS degradation as 100 healthy frames followed by 200 unhealthy frames in a loop after the initial 450 healthy frames. The specification states that the last 4 minutes (1800 frames) "may switch to an unhealthy state," but it does not specify a looping cycle.
- **Root Cause:** Hardcoded mock sequence instead of reading `server_data` or randomly triggering degradation bounds.
- **Risk Level:** **Medium** (Only affects simulation/testing, but creates false confidence).

### Issue 1.2: Optical Flow Accumulation Drift (Memory Leak / Drift)
- **Description:** In `localization.py`, the Lucas-Kanade optical flow computes translation continuously. `self._prev_points` is updated by copying new points but when point count drops below 50%, it re-detects features. Each re-detection resets the point topology, leading to discrete jumps or integration drift without a global optimization or loop closure.
- **Root Cause:** Naive frame-to-frame pixel delta summation without scaling correction over time.
- **Risk Level:** **High** (Task 2 position error will grow unbounded).

### Issue 1.3: Feature Matching Memory Leak
- **Description:** In `movement.py`, `self._tracks` caches vehicle histories. If a vehicle leaves the frame, `track.missed` increments. Tracks are deleted after `MOVEMENT_MAX_MISSED_FRAMES` (8). However, `self._cam_shift_hist` is a `deque` that safely bounds its memory, but `self._cam_total_x` and `self._cam_total_y` are floats that grow indefinitely.
- **Root Cause:** Unbounded accumulation of floats for global camera shift, which can lead to floating point precision loss over long runs.
- **Risk Level:** **Low** (Precision loss over 2250 frames is minimal, but architecturally unsafe).

### Issue 1.4: Payload Bounding Box Clamp Modifies Original Reference
- **Description:** In `network.py`, the `build_competition_payload` and `_preflight_validate_and_normalize_payload` clamp bounding boxes to `frame_w` and `frame_h`. However, if the source detection dictionary is modified in place before the JSON serialization, it corrupts the visualizer's state.
- **Root Cause:** Shared dictionary references.
- **Risk Level:** **Low** (Only affects visualization).

---

## 2. Stability Risks

- **Exception Swallowing in Detection:** `ObjectDetector.detect()` wraps the entire pipeline in a `try...except Exception as e` and returns `[]`. If an out-of-memory (OOM) error occurs or SAHI slicing fails on an edge case, the system silently reports 0 objects instead of degrading gracefully or raising an alert.
- **Feature Extraction Failure:** In `image_matcher.py` (Task 3), `cv2.findHomography` can crash or return `None` if `src_pts` and `dst_pts` contain degenerate colinear geometries. The exception is caught, but repeated failures will cause silent omission of Task 3 outputs.
- **GPU Memory Fragmentation:** `torch.cuda.empty_cache()` is called every `GPU_CLEANUP_INTERVAL` (200 frames). This is an anti-pattern that temporarily spikes CPU usage, blocks the CUDA stream, and slows down inference for the next few frames due to allocator overhead.

---

## 3. Reproducibility Risks

- **Floating Point Order in NMS:** `np.argsort()` and IoU calculations in `_nms_greedy` and `_suppress_contained` use floating point arrays. Without `stable=True` in sorting and deterministic tie-breaking rules, identical inputs on different hardware architectures (e.g., CPU vs TensorCore FP16) will yield different bounding box retention.
- **Half Precision (FP16) Variance:** The `balanced` deterministic profile leaves `HALF_PRECISION = True`. FP16 accumulation on different GPU generations (e.g., Turing vs Ampere) is non-deterministic, violating strict run-to-run equivalence. Only the `max` profile disables FP16.
- **Random Dataset Selection:** `data_loader.py` uses `random.sample()` to select DET frames. If the global seed is set *after* the loader initializes, or if the directory order varies by OS file system, the loaded images will differ across machines.

---

## 4. Performance Bottlenecks

- **Synchronous Pipeline:** `main.py` executes `fetch -> decode -> detect -> localize -> submit` in a strictly sequential loop. Network I/O latency directly blocks GPU inference. There is no background thread or `asyncio` loop for fetching the next frame while processing the current one.
- **SAHI Overhead:** `SAHI_ENABLED=True` with `SAHI_SLICE_SIZE=640` and `overlap=0.35` on a 1920x1080 frame results in at least 6 separate inference passes (1 full + ~5 sliced). On mid-range hardware, this will severely drop FPS, potentially violating the real-time constraint or causing network timeouts.
- **Duplicate Image Decoding:** In degraded network modes, the image is downloaded and decoded via `cv2.imdecode`. If fallback fails, it might be requested again or parsed redundantly.

---

## 5. Architecture Weak Points

- **Idempotency Coupling:** `network.py` uses an LRU cache (`_submitted_frames_lru`) keyed by `frame_id`. If the server drops the connection *after* receiving the payload but *before* sending the 200 OK ACK, the client marks it as failed, retries, and sends the idempotent key. But if the `frame_id` changes due to simulated clock drift or network lag, it breaks.
- **Temporal Window Lag:** `movement.py` uses a sliding window of `MOVEMENT_WINDOW_FRAMES=24` (approx 3.2 seconds at 7.5 FPS). Decisions on `motion_status` lag reality by up to 3 seconds. For fast-moving targets, this will result in incorrect motion states.
- **Config Sprawl:** `task3_params.yaml` is documented in README but never parsed or loaded in `settings.py` or `image_matcher.py`. The values are hardcoded in `Settings` class (`TASK3_SIMILARITY_THRESHOLD = 0.72`), rendering the YAML file useless and confusing.

---

## 6. Dependency Fragility

- **Ultralytics YOLO Version Coupling:** The system imports `from ultralytics import YOLO` and accesses internal attributes like `self.model.names`. If `ultralytics` updates its internal API (which happens frequently), the class mapping logic in `_configure_class_mapping` will break.
- **OpenCV Versioning:** `cv2.calcOpticalFlowPyrLK` and `cv2.SIFT_create` behavior vary across OpenCV 4.x versions. The exact OpenCV version is not pinned in this context, risking feature extraction mismatches.

---

## 7. Dead / Redundant Components

- **`MAX_BBOX_SIZE` Logic:** In `detection.py`, the `_post_filter` has logic for `MAX_BBOX_SIZE`, but it is configured to `9999`, meaning the branch is effectively dead code.
- **`task3_params.yaml`:** Discussed in architecture. Unused file.
- **`SIMULATION_DET_SAMPLE_SIZE`:** Used only in standalone testing, adds bloat to production config.
- **`UNKNOWN_OBJECTS_AS_OBSTACLES` branch:** Allows unmapped classes to invalidate landing zones, but the competition specifically expects only designated classes or explicit object bounding boxes. Since you are using a custom model that natively supports all competition classes, using unmapped or "unknown" classes as obstacles might artificially lower the mAP if the ground truth doesn't consider them, and essentially acts as dead/unnecessary logic.

---

## 8. Silent Failure Possibilities

- **JSON Payload Truncation:** If a frame contains more than `RESULT_MAX_OBJECTS` (100) detections, `_apply_object_caps` silently clips the payload and drops bounding boxes to fit the limit. If the official evaluation relies on exhaustive recall, this arbitrary clipping will silently degrade the mAP score.
- **Optical Flow Altitude Fallback:** If `server_data` returns "NaN" for `translation_z` when `gps_health=0`, `localization.py` falls back to `Settings.DEFAULT_ALTITUDE` (50.0m) to convert pixels to meters. If the actual drone altitude is 10m or 100m, the position estimate will be wildly inaccurate by a factor of 5x or 2x, respectively, without warning.
- **Missing Object ID in Task 3:** If `image_matcher.py` detects an object but the reference dictionary lacked an `object_id` during load, it defaults to a sequential integer. If the server expects specific UUIDs or hash-based IDs, the matching will be ignored.

---

## 9. Recommended Fix Strategy (No Code)

1. **Concurrency:** Implement a Producer-Consumer queue pattern using `threading` or `asyncio`. Thread 1: Fetch metadata & image. Thread 2: GPU Inference & optical flow. Thread 3: Submit JSON.
2. **Determinism Stabilization:** Disable `torch.cuda.empty_cache()` inside the inference loop. Enforce `stable=True` in numpy sorting. Move `random.seed` initialization to occur *before* any module imports.
3. **Odometry Hardening:** Implement a basic Kalman Filter or exponential moving average to smooth the Lucas-Kanade optical flow translation vectors and integrate the last known valid GPS altitude instead of a static 50m fallback.
4. **Config Cleanup:** Read `task3_params.yaml` directly into the `Settings` class dynamically or remove the YAML file entirely to maintain a single source of truth. Remove COCO/VisDrone mapping dicts as you're using a fine-tuned model.
5. **Error Propagation:** Remove naked `except Exception:` blocks in `detection.py` and `main.py`. Allow `SystemExit` or specific handled exceptions so that fatal pipeline breaks trigger the circuit breaker rather than emitting empty data.