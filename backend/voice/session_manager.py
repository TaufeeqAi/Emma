import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

from backend.config import load_tenant_config

logger = logging.getLogger(__name__)

# Maximum duration for a single call session (minutes).
# longer calls indicate the patient needs a human — escalate.
MAX_SESSION_DURATION_MINUTES = 10

# Maximum turns (patient utterances) per session.
MAX_TURNS_PER_SESSION = 20


class SessionStatus(Enum):
    CREATED = auto()
    GREETING = auto()    
    LISTENING = auto()    
    PROCESSING = auto()   
    SPEAKING = auto()     
    BARGE_IN = auto()     
    ENDED = auto()        
    ERROR = auto()        

@dataclass
class TurnRecord:
    """Record of a single patient utterance + EMMA response."""
    turn_id: str
    transcript: str
    stt_confidence: float
    escalated: bool
    final_response: str
    total_latency_ms: float
    stt_latency_ms: float
    pipeline_latency_ms: float
    tts_latency_ms: float
    timestamp: str
    error: Optional[str] = None


@dataclass
class VoiceSession:
    """
    State for a single patient call session.

    Created by SessionManager when a WebSocket connection is established.
    Destroyed when the WebSocket disconnects.

    Attributes:
        session_id:    Unique ID for this call session (UUID4).
        tenant_id:     Surgery identifier — drives all RAG isolation.
        tenant_config: Loaded surgery configuration (name, escalation msg, etc.)
        status:        Current session lifecycle status.
        turns:         Ordered list of conversation turns.
        created_at:    Session creation timestamp (UTC).
        barge_in_count: Number of times the patient interrupted EMMA.
        silence_count:  Number of silence timeouts (patient didn't respond).
    """
    session_id: str
    tenant_id: str
    tenant_config: dict
    status: SessionStatus = SessionStatus.CREATED
    turns: list = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    barge_in_count: int = 0
    silence_count: int = 0
    last_activity_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def surgery_name(self) -> str:
        return self.tenant_config.get("surgery_name", self.tenant_id)

    @property
    def escalation_message(self) -> str:
        return self.tenant_config.get(
            "escalation_message",
            "Please call 999 for emergencies or 111 for urgent advice.",
        )

    @property
    def greeting(self) -> str:
        return (
            f"Hello, you've reached {self.surgery_name}. "
            f"I'm EMMA, your AI receptionist. How can I help you today?"
        )

    @property
    def is_active(self) -> bool:
        return self.status not in (SessionStatus.ENDED, SessionStatus.ERROR)

    @property
    def is_timed_out(self) -> bool:
        """Check if the session has exceeded maximum duration."""
        created = datetime.fromisoformat(self.created_at)
        now = datetime.now(tz=timezone.utc)
        elapsed_minutes = (now - created).total_seconds() / 60
        return elapsed_minutes >= MAX_SESSION_DURATION_MINUTES

    @property
    def turn_limit_reached(self) -> bool:
        return self.turn_count >= MAX_TURNS_PER_SESSION

    def record_turn(self, turn: TurnRecord) -> None:
        """Append a completed turn to the conversation history."""
        self.turns.append(turn)
        self.last_activity_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "Session %s | Turn %d | escalated=%s | latency=%.0fms | "
            "stt_conf=%.3f | response='%.60s...'",
            self.session_id, self.turn_count, turn.escalated,
            turn.total_latency_ms, turn.stt_confidence, turn.final_response,
        )

    def record_barge_in(self) -> None:
        """Record a barge-in event."""
        self.barge_in_count += 1
        self.last_activity_at = datetime.now(tz=timezone.utc).isoformat()

    def record_silence(self) -> None:
        """Record a silence timeout event."""
        self.silence_count += 1
        self.last_activity_at = datetime.now(tz=timezone.utc).isoformat()

    def end(self, status: SessionStatus = SessionStatus.ENDED) -> None:
        """Finalise session. Called on WebSocket disconnect or timeout."""
        self.status = status
        logger.info(
            "Session %s ended | status=%s | turns=%d | "
            "barge_ins=%d | silence_events=%d",
            self.session_id, status.name, self.turn_count,
            self.barge_in_count, self.silence_count,
        )

    def to_dict(self) -> dict:
        """Serialise session for logging / Langfuse export (Phase 4)."""
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "surgery_name": self.surgery_name,
            "status": self.status.name,
            "turn_count": self.turn_count,
            "barge_in_count": self.barge_in_count,
            "silence_count": self.silence_count,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "transcript": t.transcript,
                    "escalated": t.escalated,
                    "total_latency_ms": t.total_latency_ms,
                    "stt_confidence": t.stt_confidence,
                }
                for t in self.turns
            ],
        }


class SessionManager:
    """
    Process-level registry of active VoiceSession instances.

    Allows the health endpoint and admin APIs to inspect active sessions
    without accessing WebSocket state directly.

    Thread safety: SessionManager uses a plain dict. In Phase 5 with
    multiple uvicorn workers, replace with Redis-backed session storage.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}

    def create_session(self, tenant_id: str) -> VoiceSession:
        """
        Create and register a new voice session.

        Args:
            tenant_id: Surgery identifier (validated before reaching here).

        Returns:
            New VoiceSession with loaded tenant config.

        Raises:
            FileNotFoundError: if tenant config doesn't exist.
        """
        tenant_config = load_tenant_config(tenant_id)
        session = VoiceSession(
            session_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            tenant_config=tenant_config,
        )
        self._sessions[session.session_id] = session
        logger.info(
            "Session created | session_id=%s | tenant=%s | surgery='%s'",
            session.session_id, tenant_id, session.surgery_name,
        )
        return session

    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            session.end()
        else:
            logger.warning("end_session: unknown session_id=%s", session_id)

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    @property
    def active_sessions_summary(self) -> list[dict]:
        return [
            {
                "session_id": s.session_id,
                "tenant_id": s.tenant_id,
                "turn_count": s.turn_count,
                "status": s.status.name,
                "created_at": s.created_at,
            }
            for s in self._sessions.values()
        ]