import logging
import threading
import time

import numpy as np

from .activity import ActivityRecorder, Event
from .document_processor import (
    DocumentProcessor,
    FailureKind,
    FailureReport,
    SaveMessage,
)
from .file_storage import FileStorage
from .jobs import Job, JobStore
from .storage import Storage

logger = logging.getLogger(__name__)

# Longest the worker sleeps without a wake-up, in case a job was added by another process
IDLE_SECONDS = 60


def floats(raw: bytes | None) -> np.ndarray | None:
    return None if raw is None else np.frombuffer(raw, dtype=np.float32)


class JobWorker:
    """Indexes files that were stored by async uploads, one at a time.

    The model takes one call at a time anyway, so more threads would only compete for the same cores
    """

    def __init__(
        self,
        jobs: JobStore,
        processor: DocumentProcessor,
        file_storage: FileStorage,
        storage: Storage,
        activity: ActivityRecorder,
    ) -> None:
        self.jobs = jobs
        self.processor = processor
        self.file_storage = file_storage
        self.storage = storage
        self.activity = activity
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="job-worker", daemon=True
        )

    def start(self) -> None:
        requeued = self.jobs.requeue_running()
        if requeued:
            logger.warning(
                "%d jobs were interrupted by a restart, running again", requeued
            )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        # A file in the middle of the model call is finished first, it stays running if this times out
        # and is put back on the next start
        self._thread.join(timeout=30)

    def wake(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            # Cleared before the lookup, so a job added meanwhile is not slept through
            self._wake.clear()
            job = self.jobs.claim()
            if job is None:
                self._wake.wait(IDLE_SECONDS)
                continue
            self._process(job)

    def _process(self, job: Job) -> None:
        started = time.time()
        try:
            descriptor = self.file_storage.descriptor(job.id)
            message = SaveMessage(
                original_name=descriptor.original_name,
                data=iter(()),
                content_type=descriptor.content_type,
                text_description=job.description,
                embeddings=floats(job.embeddings),
                text_embeddings=floats(job.text_embeddings),
            )
            # A job interrupted by a restart may have reached the vector storage already
            self.storage.remove([job.id])
            fails = self.processor.index([descriptor], [message])
        except Exception as e:
            logger.exception("job %s failed", job.id)
            fails = [
                FailureReport(
                    item_id=job.id,
                    kind=FailureKind.FILE_STORAGE,
                    value=f"{job.original_name}: {e}",
                )
            ]
            try:
                self.processor.rollback(fails)
            except Exception:
                logger.exception("job %s: rollback failed", job.id)

        error = (fails[0].kind, fails[0].value) if fails else None
        self.jobs.finish(job.id, error)

        code = "done" if error is None else error[0].name.lower()
        self.activity.record(
            Event(
                ts=time.time(),
                transport="worker",
                action="job_index",
                code=code,
                # A rejected file is the client's doing, a broken model or storage is ours
                outcome="ok"
                if error is None
                else "client_error"
                if error[0] is FailureKind.LIMIT_EXCEEDED
                else "server_error",
                user_id=job.owner_id,
                actor="worker",
                duration_ms=(time.time() - started) * 1000,
                detail=None if error is None else error[1],
            )
        )
