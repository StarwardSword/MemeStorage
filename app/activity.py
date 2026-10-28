import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from .configuration import ActivityConfiguration

logger = logging.getLogger(__name__)

Outcome = Literal["ok", "denied", "client_error", "server_error"]

BATCH = 200
FLUSH_SECONDS = 1.0
PRUNE_SECONDS = 3600
QUEUE_SIZE = 10_000


def outcome_of(code: str) -> Outcome:
    """HTTP status or gRPC status name to one of four outcomes"""
    if code in ("401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED"):
        return "denied"
    if code.isdigit():
        n = int(code)
        return "ok" if n < 400 else "client_error" if n < 500 else "server_error"
    if code == "OK":
        return "ok"
    if code in ("INTERNAL", "UNKNOWN", "UNAVAILABLE", "DATA_LOSS", "UNIMPLEMENTED"):
        return "server_error"
    return "client_error"


class Event(BaseModel):
    ts: float
    transport: Literal["http", "grpc", "worker"]
    # Name of the handler or the RPC: save_file, search_files, Save, GetByQuery, job_index
    action: str
    # HTTP status or gRPC status name
    code: str
    outcome: Outcome
    user_id: uuid.UUID | None = None
    # Name of the user at that time, "admin", "worker" or None when nobody was identified
    actor: str | None = None
    duration_ms: float = 0
    bytes_in: int = 0
    bytes_out: int = 0
    ip: str | None = None
    # URL path for HTTP, never the query string or the token
    path: str | None = None
    # What is worth knowing besides that: text of a search query
    detail: str | None = None


class ActionStats(BaseModel):
    action: str
    requests: int
    ok: int
    denied: int
    client_error: int
    server_error: int
    avg_ms: float
    p95_ms: float
    max_ms: float
    bytes_in: int
    bytes_out: int


class ActorStats(BaseModel):
    user_id: uuid.UUID | None
    actor: str | None
    requests: int
    errors: int
    bytes_in: int
    bytes_out: int
    first_seen: float
    last_seen: float


class DayStats(BaseModel):
    day: str
    requests: int
    errors: int
    bytes_in: int
    bytes_out: int


class QueryCount(BaseModel):
    query: str
    count: int


class IpStats(BaseModel):
    ip: str
    requests: int
    denied: int


class JobCounts(BaseModel):
    by_status: dict[str, int] = {}
    # Why failed jobs failed
    failures: dict[str, int] = {}


class ActivitySummary(BaseModel):
    since: float
    until: float
    requests: int
    ok: int
    denied: int
    client_error: int
    server_error: int
    avg_ms: float
    bytes_in: int
    bytes_out: int
    by_transport: dict[str, int]
    by_action: list[ActionStats]
    by_actor: list[ActorStats]
    by_day: list[DayStats]
    top_queries: list[QueryCount]
    by_ip: list[IpStats]
    jobs: JobCounts = JobCounts()
    # Events lost because the writer could not keep up, see ActivityRecorder
    dropped: int = 0


class ActivityStore(ABC):
    @abstractmethod
    def record(self, events: list[Event]) -> None:
        raise NotImplementedError

    @abstractmethod
    def events(
        self,
        since: float,
        until: float,
        user_id: uuid.UUID | None,
        action: str | None,
        outcome: str | None,
        limit: int,
        offset: int,
    ) -> list[Event]:
        """Newest first"""
        raise NotImplementedError

    @abstractmethod
    def summary(self, since: float, until: float) -> ActivitySummary:
        raise NotImplementedError

    @abstractmethod
    def prune(self, before: float) -> int:
        raise NotImplementedError


class ActivityRecorder:
    """Takes events from request threads and writes them in batches from its own thread.

    Recording must never slow a request down or fail it, so a full queue drops the event
    and counts it instead of waiting
    """

    def __init__(self, store: ActivityStore | None, config: ActivityConfiguration):
        self.store = store if config.enabled else None
        self.retention_days = config.retention_days
        self.record_queries = config.record_queries
        self.dropped = 0
        self._queue: queue.Queue[Event] = queue.Queue(QUEUE_SIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="activity-writer", daemon=True
        )

    @property
    def enabled(self) -> bool:
        return self.store is not None

    def start(self) -> None:
        if self.store is not None:
            self._thread.start()

    def stop(self) -> None:
        if self.store is None:
            return
        self._stop.set()
        self._thread.join(timeout=10)

    def record(self, event: Event) -> None:
        if self.store is None:
            return
        if not self.record_queries:
            event.detail = None
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def _drain(self, wait: float) -> list[Event]:
        batch: list[Event] = []
        try:
            batch.append(self._queue.get(timeout=wait))
            while len(batch) < BATCH:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return batch

    def _write(self, batch: list[Event]) -> None:
        if not batch or self.store is None:
            return
        try:
            self.store.record(batch)
        except Exception:
            logger.exception("activity: %d events were lost", len(batch))

    def _run(self) -> None:
        pruned_at = 0.0
        while not self._stop.is_set():
            self._write(self._drain(FLUSH_SECONDS))
            if time.time() - pruned_at > PRUNE_SECONDS and self.store is not None:
                pruned_at = time.time()
                try:
                    self.store.prune(time.time() - self.retention_days * 86400)
                except Exception:
                    logger.exception("activity: pruning failed")

        self.flush()

    # Writes what is queued now, from the calling thread
    def flush(self) -> None:
        while batch := self._drain(0.0):
            self._write(batch)
