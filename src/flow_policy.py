"""Runtime flow/state policy helpers for competition loop decisions."""

from dataclasses import dataclass
from enum import Enum


class FrameLifecycleState(str, Enum):
    IDLE = "IDLE"
    FETCHED = "FETCHED"
    PROCESSED = "PROCESSED"
    SUBMITTING = "SUBMITTING"
    ACKED = "ACKED"
    TERMINAL = "TERMINAL"


class FetchStrategy(str, Enum):
    FULL_FRAME = "full_frame"
    FALLBACK_ONLY = "fallback_only"


class DuplicateStormAction(str, Enum):
    TERMINATE_SESSION = "terminate_session"
    CONTINUE = "continue"


@dataclass(frozen=True)
class DegradeFetchDecision:
    strategy: FetchStrategy
    reason_code: str
    degrade_seq: int


def decide_degrade_fetch_strategy(
    is_degraded: bool,
    degrade_seq: int,
    heavy_every: int,
    force_full_frame: bool = False,
) -> DegradeFetchDecision:
    if not is_degraded:
        return DegradeFetchDecision(
            strategy=FetchStrategy.FULL_FRAME,
            reason_code="normal_mode",
            degrade_seq=max(0, int(degrade_seq)),
        )

    slot = max(1, int(degrade_seq))
    if force_full_frame:
        return DegradeFetchDecision(
            strategy=FetchStrategy.FULL_FRAME,
            reason_code="degrade_recovery_heavy_forced",
            degrade_seq=slot,
        )

    interval = max(1, int(heavy_every))
    if slot % interval == 0:
        return DegradeFetchDecision(
            strategy=FetchStrategy.FULL_FRAME,
            reason_code="degrade_heavy_slot",
            degrade_seq=slot,
        )

    return DegradeFetchDecision(
        strategy=FetchStrategy.FALLBACK_ONLY,
        reason_code="degrade_fetch_only_slot",
        degrade_seq=slot,
    )


def decide_duplicate_storm_action(
    consecutive_duplicates: int,
    threshold: int,
    configured_action: str,
) -> DuplicateStormAction:
    if int(consecutive_duplicates) < max(1, int(threshold)):
        return DuplicateStormAction.CONTINUE

    normalized = str(configured_action or "").strip().lower()
    if normalized == DuplicateStormAction.TERMINATE_SESSION.value:
        return DuplicateStormAction.TERMINATE_SESSION
    return DuplicateStormAction.CONTINUE
