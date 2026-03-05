"""Retry and idempotency guard helpers for frame result submission."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SubmitAttemptGuard:
    """Tracks in-flight/acked frame keys to prevent duplicate sends for same frame."""

    max_size: int = 512
    _in_flight: "OrderedDict[str, None]" = field(default_factory=OrderedDict)
    _acked: "OrderedDict[str, None]" = field(default_factory=OrderedDict)

    def should_block_new_send(self, frame_key: str) -> bool:
        return frame_key in self._in_flight

    def is_already_acked(self, frame_key: str) -> bool:
        if frame_key in self._acked:
            self._acked.move_to_end(frame_key)
            return True
        return False

    def mark_in_flight(self, frame_key: str) -> None:
        self._touch(self._in_flight, frame_key)

    def mark_acked(self, frame_key: str) -> None:
        self._in_flight.pop(frame_key, None)
        self._touch(self._acked, frame_key)

    def clear_in_flight(self, frame_key: str) -> None:
        self._in_flight.pop(frame_key, None)

    def _touch(self, lru: "OrderedDict[str, None]", key: str) -> None:
        lru[key] = None
        lru.move_to_end(key)
        while len(lru) > max(1, int(self.max_size)):
            lru.popitem(last=False)


def build_idempotency_key(prefix: str, session_id: str, run_uuid: str, frame_key: str) -> str:
    return f"{prefix}:{session_id}:{run_uuid}:{frame_key}"
