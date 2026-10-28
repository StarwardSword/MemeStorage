import uuid
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from .activity import JobCounts
from .document_processor import FailureKind


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    # Same as the id of the stored file
    id: uuid.UUID
    original_name: str
    description: str | None = None
    # Uploader of the file, for statistics. None for the admin
    owner_id: uuid.UUID | None = None
    # float32 bytes of the ready embeddings the client sent, if any
    embeddings: bytes | None = None
    text_embeddings: bytes | None = None
    status: JobStatus = JobStatus.PENDING
    error_kind: FailureKind | None = None
    error_message: str | None = None


class JobStore(ABC):
    @abstractmethod
    def add(self, jobs: list[Job]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, id: uuid.UUID) -> Job | None:
        raise NotImplementedError

    @abstractmethod
    def claim(self) -> Job | None:
        """Takes the oldest pending job and marks it running"""
        raise NotImplementedError

    @abstractmethod
    def finish(self, id: uuid.UUID, error: tuple[FailureKind, str] | None) -> None:
        raise NotImplementedError

    @abstractmethod
    def active_count(self) -> int:
        """Pending and running jobs"""
        raise NotImplementedError

    @abstractmethod
    def counts(self) -> JobCounts:
        raise NotImplementedError

    @abstractmethod
    def requeue_running(self) -> int:
        """Puts back jobs that were running when the server stopped"""
        raise NotImplementedError
