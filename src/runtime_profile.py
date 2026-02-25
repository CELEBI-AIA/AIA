"""Runtime profile and determinism helpers."""

import os
import random
from typing import Literal, Optional

import numpy as np
import torch

from config.settings import Settings
from src.utils import Logger

ProfileName = Literal["off", "balanced", "max"]


def apply_runtime_profile(profile: ProfileName, requested_profile: Optional[str] = None) -> None:
    """Apply deterministic/runtime behavior profile at startup."""
    log = Logger("Runtime")
    requested = (requested_profile or profile or "balanced").strip().lower()
    profile = (profile or "balanced").strip().lower()

    if profile not in {"off", "balanced", "max"}:
        raise ValueError(f"Unsupported deterministic profile: {profile}")

    if profile == "off":
        log.info(
            f"Deterministic profile applied | requested={requested} | effective=off"
        )
        return

    seed = int(Settings.DETERMINISM_SEED)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    cpu_threads = max(1, int(Settings.DETERMINISM_CPU_THREADS))
    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass

    # Profile-specific runtime toggles.
    if profile in {"balanced", "max"}:
        Settings.AUGMENTED_INFERENCE = False
    if profile == "max":
        Settings.HALF_PRECISION = False

    log.success(
        f"Deterministic profile applied | requested={requested} | effective={profile} | "
        f"seed={seed} | tta={'off' if not Settings.AUGMENTED_INFERENCE else 'on'} | "
        f"fp16={'on' if Settings.HALF_PRECISION else 'off'}"
    )
