"""Session-loop state machine for frame receive/infer/send/ack orchestration."""

from dataclasses import dataclass
from enum import Enum


class SessionFlowState(str, Enum):
    IDLE = "IDLE"
    RECEIVED = "RECEIVED"
    INFERRED = "INFERRED"
    SENT = "SENT"
    ACKED = "ACKED"
    TERMINAL = "TERMINAL"


@dataclass
class SessionFlowMachine:
    """Single checkpoint controller that blocks fetch until ACK is observed."""

    state: SessionFlowState = SessionFlowState.IDLE

    _ALLOWED = {
        SessionFlowState.IDLE: {SessionFlowState.RECEIVED, SessionFlowState.TERMINAL},
        SessionFlowState.RECEIVED: {SessionFlowState.INFERRED, SessionFlowState.TERMINAL},
        SessionFlowState.INFERRED: {SessionFlowState.SENT, SessionFlowState.TERMINAL},
        SessionFlowState.SENT: {SessionFlowState.ACKED, SessionFlowState.TERMINAL},
        SessionFlowState.ACKED: {SessionFlowState.IDLE, SessionFlowState.TERMINAL},
        SessionFlowState.TERMINAL: {SessionFlowState.TERMINAL},
    }

    def transition(self, new_state: SessionFlowState) -> None:
        if new_state not in self._ALLOWED.get(self.state, set()):
            raise ValueError(
                f"Invalid session flow transition: {self.state.value} -> {new_state.value}"
            )
        self.state = new_state

    def can_fetch_new_frame(self) -> bool:
        """Single fetch gate: only IDLE can fetch a new frame.

        ACKED must explicitly transition back to IDLE before fetch is allowed.
        """

        return self.state == SessionFlowState.IDLE
