import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

MAX_HISTORY: int = 200
HEARTBEAT_INTERVAL: int = 15  # seconds


class EventBus:
    """
    Fan-out event bus: one publisher → many SSE subscribers.

    Usage:
        bus = get_event_bus()

        # Publisher (in LiveKit adapter, webhook handler, etc.)
        bus.publish({"type": "transcript", "room_name": "...", "text": "..."})

        # Subscriber (in SSE endpoint)
        async for event in bus.subscribe():
            yield f"data: {json.dumps(event)}\\n\\n"
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._history: list[dict] = []
        self._heartbeat_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start heartbeat task. Call from FastAPI lifespan startup."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="sse_heartbeat"
            )
            logger.info("EventBus started (heartbeat every %ds)", HEARTBEAT_INTERVAL)

    def stop(self) -> None:
        """Stop heartbeat task. Call from FastAPI lifespan shutdown."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

    def publish(self, event: dict) -> None:
        """
        Publish an event to all active subscribers.

        Non-blocking: uses put_nowait(). Slow subscribers whose queues are
        full are automatically dropped (their connection will timeout and
        browser EventSource will reconnect with ?lastEventId).
        """
        if "timestamp" not in event:
            event = {**event, "timestamp": _now_iso()}

        # Append to ring buffer
        self._history.append(event)
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]

        # Fan-out to subscribers, drop dead queues
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "EventBus: subscriber queue full — dropping event type=%s",
                    event.get("type"),
                )
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    async def subscribe(
        self, room_name: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields events as they arrive.

        Args:
            room_name: If set, yields only events for that room_name
                       (plus roomless events like heartbeats).

        Yields recent history first (last 50 events), then live events.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=150)
        self._subscribers.append(q)
        try:
            # Replay recent history to the new subscriber
            history_slice = self._history[-50:]
            for event in history_slice:
                if _matches_filter(event, room_name):
                    yield event

            # Then stream live events
            while True:
                event = await q.get()
                if _matches_filter(event, room_name):
                    yield event
        except asyncio.CancelledError:
            pass
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)

    @property
    def history(self) -> list[dict]:
        """Returns a copy of the event history."""
        return list(self._history)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def clear_history(self) -> None:
        self._history.clear()

    # ── Private 

    async def _heartbeat_loop(self) -> None:
        """Emit a heartbeat event every HEARTBEAT_INTERVAL seconds."""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                self.publish({"type": "heartbeat", "timestamp": _now_iso()})
        except asyncio.CancelledError:
            pass


# ── Helpers 

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _matches_filter(event: dict, room_name: Optional[str]) -> bool:
    """Return True if the event should be delivered to this subscriber."""
    if room_name is None:
        return True
    event_room = event.get("room_name")
    # Deliver roomless events (heartbeat, etc.) to all
    if event_room is None:
        return True
    return event_room == room_name


# ── Module-level singleton 

_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus