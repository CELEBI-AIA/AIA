==================================================
PROJECT STATE REPORT
Generated At: 2026-02-22 18:37:52 +03
==================================================

1) Project Identity
- Project name (detectable): AIA (repository under `HavaciliktaYapayZeka/AIA`).
- Primary purpose: TEKNOFEST competition runtime for per-frame object detection, landing suitability signaling, and position estimation under GPS degradation.
- Competition/task alignment: Targets Task-1 (object classes + landing suitability) and Task-2 (position estimation) with server-connected competition mode and offline simulation modes.
- Main problem it solves: Consumes ordered frames, runs local inference/estimation, and returns structured JSON results in a constrained offline-like environment.

2) Repository Overview
- Directory structure summary:
  - `main.py`: runtime entry point and loop orchestration.
  - `src/`: detection, localization, network, movement, runtime profile, utilities.
  - `config/`: centralized settings + task3 params document.
  - `requirements.txt`: now pinned dependency versions.
  - `CODEBASE_STATE_REPORT.md`: current engineering state report.
- Core modules:
  - `src/detection.py`: YOLO-based detection, SAHI slicing, landing suitability rules.
  - `src/movement.py`: centroid-history movement labeling (internal runtime use).
  - `src/localization.py`: GPS + optical-flow hybrid odometry.
  - `src/network.py`: fetch-state model, retries, payload building, bbox clamping.
  - `src/runtime_profile.py`: determinism/runtime profile bootstrap.
  - `src/utils.py`: logging, visualizer, JSON log persistence/rotation.
- Entry points:
  - `main.py` (CLI-first): `--mode competition|simulate_vid|simulate_det`, optional `--interactive`.
- Config files:
  - `config/settings.py` (single runtime config source).
  - `config/task3_params.yaml` (threshold file, not integrated into active runtime flow).
- Model files:
  - Runtime expects local `models/yolov8m.pt` path.
- Utilities:
  - `Visualizer`, sampled JSON logging, and bounded log-pruning in utility layer.

3) System Architecture (Current Implementation)
- Actual implemented architecture:
  - Single-process synchronous pipeline coordinated by `main.py`.
  - Competition loop now uses explicit frame-fetch state machine (`OK`, `END_OF_STREAM`, `TRANSIENT_ERROR`, `FATAL_ERROR`).
- Data flow from input to output:
  1. Startup applies runtime profile via `apply_runtime_profile()`.
  2. `NetworkManager.get_frame()` retrieves metadata with typed status.
  3. `download_image()` obtains and decodes frame bytes.
  4. `ObjectDetector.detect()` performs preprocess -> inference -> filtering -> landing-state assignment.
  5. `MovementEstimator.annotate()` tags movement internally.
  6. `VisualOdometry.update()` updates translation estimate.
  7. `send_result()` builds specification-aligned payload and POSTs with retry.
- Detection pipeline:
  - CLAHE + unsharp preprocessing.
  - Full-frame + optional sliced inference (SAHI), then NMS/containment suppression.
  - Post-filters for bbox size/aspect.
- Tracking/motion logic:
  - Per-vehicle center tracking with nearest-match gating and history displacement threshold.
- Landing suitability logic:
  - Vehicles/humans -> `-1`.
  - UAP/UAI -> edge-touch and overlap-based suitability (`0` or `1`).
- Position estimation logic:
  - GPS healthy: direct telemetry update.
  - GPS unhealthy: Lucas-Kanade optical flow with feature refresh policy.
- JSON output generation:
  - Payload builder enforces field types and clamps bboxes to frame bounds.
  - Outbound payload includes top-level `id`, `user`, `frame`, per-object `motion_status`, and `detected_undefined_objects`.
- Server communication flow:
  - Session handshake -> frame loop -> robust transient/fatal handling -> bounded backoff on transient errors.

4) Implemented Features (What Actually Exists)
- Object detection: Implemented (YOLO + SAHI + post-filters).
- Motion classification:
  - Implemented internally for vehicle tracks.
  - Transmitted to server payload as per-object `motion_status` (`-1/0/1`).
- Landing logic: Implemented in detector post-processing path.
- Position estimation: Implemented (GPS + optical flow hybrid).
- Tracking: Implemented lightweight tracker (movement-only concern).
- Error handling:
  - Improved in network layer via typed fetch states and transient budget handling.
  - Broad exception guards remain in runtime loop.
- Logging:
  - Structured leveled logger.
  - JSON logs with file-name sanitization and retention pruning (`LOG_MAX_FILES`).
- Config management: Centralized in `settings.py`; runtime profile mutates selected settings at startup.
- Determinism control:
  - Newly implemented profile-based control (`off`, `balanced`, `max`).
  - Seeds set for Python/NumPy/Torch; deterministic backend flags applied; TTA forced off in deterministic profiles.
- Offline safeguards:
  - Local model loading and no cloud SDK dependencies in runtime path.
  - No hard allowlist enforcing local-only host targets.

Implementation status summary:
- Fully implemented: CLI-first startup, resilient fetch-state network loop, specification-aligned payload sanitation/clamp, deterministic profile bootstrap.
- Partially implemented: comprehensive determinism (FP16 still on in balanced profile), explicit offline endpoint enforcement.
- Missing/effectively constrained: explicit pipeline producing UAP/UAI classes remains dependent on model/class map capability (no dedicated UAP/UAI detector path).

5) Determinism & Reproducibility Status
- Seed usage:
  - Implemented in `runtime_profile.py` (`PYTHONHASHSEED`, `random`, `numpy`, `torch`, `torch.cuda`).
- Random components:
  - Simulation dataset selection still uses randomness (seeded when deterministic profiles are active).
- Multithreading risks:
  - Runtime mostly single-thread loop; CPU thread count now configurable (`DETERMINISM_CPU_THREADS`).
- GPU nondeterminism risks:
  - `torch.use_deterministic_algorithms(True, warn_only=True)` and `cudnn.deterministic=True`, `benchmark=False` are set in deterministic profiles.
  - Balanced profile keeps FP16 enabled, leaving residual numeric variance risk.
- Order sensitivity risks:
  - NMS and candidate sorting remain floating-point/order sensitive in edge ties.
- Floating point stability risks:
  - Optical flow integration drift and FP16 rounding remain potential variation sources.

Assessment: determinism posture materially improved versus prior state, but not absolute in balanced profile.

6) Performance Characteristics (Based on Code)
- Frame processing structure: strict serial processing per frame.
- Per-frame computation flow: detector dominates compute; SAHI magnifies cost by tiled passes.
- Memory handling approach:
  - Periodic CUDA cache emptying in detection.
  - Image caching in simulation path.
- Potential bottlenecks:
  - SAHI tile inference loops.
  - Blocking HTTP and decode operations.
  - Debug visualization and log I/O when enabled.
- Blocking operations:
  - All network calls remain synchronous.
- I/O strategy:
  - In-memory decode; sampled JSON logs; pruning avoids unbounded log growth.
- Scalability risks:
  - No async pipelining or batching.
  - High-resolution + SAHI + constrained hardware can lower usable FPS.

7) Competition Compliance Check
- Class ID mapping correctness:
  - Vehicle/human mappings present and active.
  - UAP/UAI production remains unresolved at model mapping level in current detector implementation (no direct COCO mapping entries for cls 2/3).
- Landing suitability rule correctness:
  - Implemented logic matches stated rule semantics (edge completeness + overlap invalidation).
- Motion classification logic correctness:
  - Implemented internally; currently not part of outbound payload by design.
- JSON format compliance:
  - Specification-aligned payload shape is enforced consistently (`id/user/frame/detected_objects/detected_translations/detected_undefined_objects`).
  - Per-object movement signal is emitted as `motion_status`.
- Offline compliance (internet usage detection):
  - No external APIs in code path.
  - No runtime guard that rejects non-local internet base URLs.
- Hardcoded assumptions:
  - Static focal length fallback and threshold constants remain.

8) Code Quality Assessment
- Modularity: Good separation across runtime modules.
- Separation of concerns: Improved after adding `runtime_profile.py` and stricter `network.py` responsibilities.
- Config centralization: Strong; most runtime knobs in `settings.py`.
- Magic numbers: Still present in detection/localization thresholds and margins.
- Dependency hygiene: Improved via pinned `requirements.txt`.
- Readability: High; descriptive naming and substantial inline documentation.
- Technical debt zones:
  - README and implementation can still drift (historical evidence).
  - Task-3 param file remains documented but not wired into runtime execution.

9) Risk Zones
- Architectural risks:
  - Serial pipeline cannot mask network latency or inference jitter.
- Runtime risks:
  - Broad exception swallowing may reduce failure visibility under recurring bad inputs.
- Performance risks:
  - SAHI + high image size can exceed practical competition throughput margins.
- Determinism risks:
  - Balanced profile still allows FP16-induced minor variation.
- Competition failure risks:
  - If UAP/UAI classes are not generated by model/mapping, landing scoring capability remains structurally limited.

10) Missing Components
- To be production-ready:
  - Automated integration tests for fetch states, payload schema, bbox clamp, and odometry transitions.
  - Runtime observability metrics (latency histograms, per-stage timings, error cardinality).
- To be competition-ready:
  - Verified end-to-end UAP/UAI class output path on target competition datasets/videos.
  - Contract test harness against official server behavior and payload acceptance edge cases.
  - Optional endpoint policy guard for offline-only host targets.
- To be research-grade:
  - Formal reproducibility manifest (exact CUDA/cuDNN/driver/runtime hashes) and benchmark suite.

11) Technical Maturity Score
- Architecture maturity: 7/10
- Determinism safety: 6/10
- Performance optimization: 7/10
- Competition robustness: 5/10
- Maintainability: 8/10

12) Summary Verdict
- Is this system stable? Moderately stable for controlled runs; improved resilience under transient network failures.
- Is it fragile? Less fragile than previous revision, but still sensitive to detector/model capability limits and synchronous bottlenecks.
- Is it competition-ready? Partially; core loop quality improved, but class-output completeness for UAP/UAI remains the decisive risk.
- Biggest weakness: Competition-critical detection coverage mismatch risk (especially UAP/UAI output path), which directly impacts landing suitability scoring potential.
