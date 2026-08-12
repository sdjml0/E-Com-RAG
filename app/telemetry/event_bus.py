import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any, List, Set
from app.schemas import PipelineTelemetryEvent

logger = logging.getLogger("event_bus")

class EventBus:
    """In-memory async event bus for pub/sub microservice telemetry SSE streaming."""
    
    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(queue)
        logger.info(f"New SSE client subscribed. Total active subscribers: {len(self._subscribers)}")
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)
        logger.info(f"SSE client unsubscribed. Total active subscribers: {len(self._subscribers)}")

    async def publish(self, event_type: str, details: Dict[str, Any], trace_id: str | None = None) -> None:
        """Publish a pipeline event to all connected SSE clients."""
        event = PipelineTelemetryEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,  # "health_update", "pipeline_stage", "moving_data", "error_event"
            trace_id=trace_id,
            details=details
        )
        data = event.model_dump()
        
        # Broadcast to queues
        dead_queues = set()
        for queue in self._subscribers:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                dead_queues.add(queue)
                
        for q in dead_queues:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

# Global singleton event bus
event_bus = EventBus()
