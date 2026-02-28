# TEKNOFEST Aviation AI System Audit Report
**File:** `AIA_SYSTEM_AUDIT.md`

## 1. Executive Summary
This report presents a deep technical audit of the TEKNOFEST 2026 Aviation AI System. The system provides real-time object detection (Task 1), visual odometry (Task 2), and reference object matching (Task 3). The primary focus of this audit is to detect logical bugs, silent failures, pipeline breaks, unstable training flows, configuration mismatches, memory leaks, GPU inefficiencies, reproducibility issues, and architectural weak points across various potential competition hardware.

## 2. Detected Issues

### 2.1 Absence of Training Flow / Unstable Training Logic
*   **Description:** The entire repository is geared exclusively toward inference, data loading, networking, and resilience. There are absolutely no training scripts, validation loops, hyperparameter configurations, or data augmentation pipelines present for the YOLOv8 model or any other trainable component.
*   **Root Cause:** The project assumes a pre-trained model (`models/yolov8m.pt`) is provided and focuses solely on competition-day runtime orchestration.
*   **Risk Level:** **Critical**
*   **Recommended Fix Strategy:** Introduce a dedicated `training/` module with reproducible scripts (e.g., using `ultralytics` CLI or Python API) to train, validate, and test models. This must include explicit seed fixing, dataset splitting logic, and clear augmentation parameters.

### 2.2 Unsafe Dictionary Iteration in Tracker
*   **Description:** In `src/movement.py`'s `_age_tracks` method, there is a risk of a `RuntimeError: dictionary changed size during iteration` if logic is added that mutates `_tracks` during the track aging loop. While the current implementation safely collects keys `to_delete` first, this pattern is fragile and prone to breaking during future modifications.
*   **Root Cause:** Standard Python dictionary iteration mechanics.
*   **Risk Level:** **Low**
*   **Recommended Fix Strategy:** Ensure iteration always happens over a static copy of the keys (e.g., `list(self._tracks.keys())`) to prevent silent failures or crashes if future logic alters the dictionary in-place.

### 2.3 Unsafe Feature Extraction Fallback in Motion Compensation
*   **Description:** In `src/movement.py`, if optical flow feature detection fails (`cv2.calcOpticalFlowPyrLK` returns `None`), the code attempts a fallback `cv2.goodFeaturesToTrack`. If this *also* fails (e.g., due to extreme blur or a completely flat image), `self._prev_points` becomes `None`. However, subsequent frames assume `self._prev_points` is a valid array if its length is checked, which could raise a `TypeError: object of type 'NoneType' has no len()`.
*   **Root Cause:** Incomplete `None` checking in the optical flow fallback logic.
*   **Risk Level:** **High**
*   **Recommended Fix Strategy:** Add strict `None` guards before accessing or computing the length of `self._prev_points` and explicitly handle cases where no features can be tracked across multiple frames.

### 2.4 Silent Failure in Visual Odometry Feature Tracking
*   **Description:** In `src/localization.py`, if the optical flow loses track of points, it attempts to re-initialize reference points using `cv2.goodFeaturesToTrack`. If this function returns `None` (no corners found), the next frame will attempt to evaluate `len(self._prev_points) < 10`. Calling `len(None)` will crash the localization loop, causing a total failure of Task 2.
*   **Root Cause:** Unsafe length check on a potentially `None` object returned by OpenCV.
*   **Risk Level:** **Critical**
*   **Recommended Fix Strategy:** Implement a safe check (`if self._prev_points is None or len(self._prev_points) < 10:`) to prevent the `TypeError` and allow the system to gracefully handle featureless frames.

### 2.5 Infinite Loop Risk in Network Retry Logic (JSON Parsing)
*   **Description:** In `src/network.py`'s `get_frame()`, a `ValueError` during JSON parsing immediately breaks the retry loop and returns a `FATAL_ERROR`. If the server momentarily returns malformed JSON due to a proxy issue or temporary corruption, the system will irreversibly drop the frame rather than retrying, potentially causing a desync with the server's expected frame sequence.
*   **Root Cause:** Hard failure on JSON decode errors instead of treating them as transient network issues.
*   **Risk Level:** **Medium**
*   **Recommended Fix Strategy:** Catch `ValueError` or `requests.exceptions.JSONDecodeError` and treat it as a transient error, triggering the standard backoff and retry logic rather than an immediate fatal return.

## 3. Stability Risks
*   **OOM Vulnerability:** While `src/detection.py` catches `torch.cuda.OutOfMemoryError`, recovering from an OOM during inference can leave PyTorch in an unstable state. Continuous OOMs on high-resolution images (especially if `SAHI_ENABLED` is true and slicing creates massive batches) will thrash the GPU, destroying performance.
*   **Circuit Breaker Flapping:** The `SessionResilienceController` in `src/resilience.py` prevents network storms, but aggressive degradation (`DEGRADE_SEND_INTERVAL_FRAMES = 3`) might cause the system to oscillate wildly between normal and degraded states if the network latency hovers exactly around the timeout threshold.

## 4. Reproducibility Risks
*   **Missing Environment Pinning:** The repository uses `requirements.txt` but relies on dynamic downloads of PyTorch via the `--index-url` argument in the README. There is no strict hash-checking or environment locking (like `poetry.lock` or `Pipfile.lock`), meaning deployment on competition day might fetch newer, incompatible versions of dependencies like `ultralytics` or `opencv-python`.
*   **Hardware-Dependent Determinism:** While `src/runtime_profile.py` sets manual seeds for `torch`, `numpy`, and `random`, GPU operations (especially with cuDNN) can still exhibit non-deterministic behavior across different GPU architectures (e.g., RTX 3060 vs. T4 vs. A100) even when `deterministic = True`. 

## 5. Performance Bottlenecks
*   **Synchronous Inference Loop:** The main loop in `main.py` performs network fetching, heavy inference (`YOLO`, `SAHI`), optical flow (`localization.py`), feature matching (`image_matcher.py`), and network submission sequentially. This completely stalls the pipeline and prevents overlapping network I/O with GPU computation.
*   **Redundant Feature Extraction:** `src/localization.py` and `src/movement.py` both compute optical flow/corner detection independently on the same gray-scale frames. This duplicate processing wastes CPU cycles.

## 6. Architecture Weak Points
*   **Idempotency Key Collision:** The `NetworkManager` generates idempotency keys based solely on `frame_id`. If the system crashes and restarts, or if the server re-issues a frame ID due to a reset, the client might mistakenly drop valid frames via its LRU cache or the server might reject them if the key doesn't include a session/run identifier.
*   **Hardcoded File Paths:** Settings like `TEMP_FRAME_PATH` and model paths are deeply coupled to the repository structure. Running the script from a different working directory other than the repo root may cause unexpected `FileNotFoundError`s.

## 7. Dependency Fragility
*   **OpenCV GUI Dependencies:** The code imports `cv2` and utilizes `cv2.imshow`, which relies on underlying GUI libraries (like `libgl1-mesa-glx` on Linux). If the competition server or headless docker container lacks these, the entire script will crash on import, even if `show=False`.
*   **Requests Library Timeouts:** The timeout logic in `src/network.py` is robust, but relies on the underlying OS socket behavior for DNS resolution, which isn't always fully respected by the `requests` library timeout parameter, potentially causing long hangs.

## 8. Dead / Redundant Components
*   **Task 3 Config File:** The README notes that `config/task3_params.yaml` is deprecated and parameters are now in `settings.py`. The presence of the dead YAML file creates configuration confusion and is technically dead code.
*   **Static Simulation Image:** The system utilizes a hardcoded `bus.jpg` for pure simulation mode, which does not adequately test the temporal mechanics (optical flow, movement tracking) that rely on sequential frames.

## 9. Silent Failure Possibilities
*   **Empty Bounding Boxes:** If SAHI or standard inference produces a bounding box where `x1 == x2` or `y1 == y2`, the area becomes zero. This could cause `ZeroDivisionError` or silent math failures in NMS or overlap calculations if not perfectly guarded by `max(..., 1e-6)`.
*   **Homography Degeneracy:** In `src/image_matcher.py`, finding a homography matrix can silently return `None` or an invalid matrix if the matched points are collinear, leading to dropped reference objects without clear warnings.

## 10. Global Recommended Fix Strategy
1.  **Pipeline Asynchrony:** Refactor the main orchestrator (`main.py`) to decouple network I/O from GPU computation using Python's `asyncio` or `concurrent.futures.ThreadPoolExecutor`. This will hide network latency and maximize GPU utilization.
2.  **Strict Environment Management:** Replace `requirements.txt` with a locked dependency manager (e.g., Poetry) that pins exact versions and hashes for `torch`, `torchvision`, `ultralytics`, and `opencv-python-headless` (to avoid GUI crashes).
3.  **Shared Computation:** Introduce a centralized "Frame Context" object that computes grayscale conversions, basic feature points, and optical flow once per frame, passing the results to both `movement.py` and `localization.py` to eliminate redundant work.
4.  **Null-Safety Audit:** Conduct a thorough pass on all OpenCV calls (especially `goodFeaturesToTrack` and `calcOpticalFlowPyrLK`) to strictly validate returned shapes and handle `None` values to prevent runtime crashes (specifically `TypeError: object of type 'NoneType' has no len()`).
5.  **Establish Training Scaffold:** Create a distinct `training/` pipeline separate from runtime inference to formalize model retraining, ensuring data augmentation and deterministic validation flows are documented and preserved.
