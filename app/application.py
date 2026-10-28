import hmac
import time
import uuid
from collections.abc import Callable
from threading import Lock
from typing import Iterator

import numpy as np

from .activity import (
    ActivityRecorder,
    ActivityStore,
    ActivitySummary,
    Event,
)
from .auth import (
    USER_NAME,
    Principal,
    User,
    UserStore,
    hash_token,
    new_token,
)
from .configuration import ActivityConfiguration, AuthConfiguration, Configuration
from .document_processor import DocumentProcessor, FailureReport, SaveMessage
from .embeddings import (
    EmbeddingProvider,
    Embeddings,
    WrongEmbeddingsError,
)
from .file_storage import FileStorage
from .file_storage_models import FileDescriptor, FileRecord, StoredFile
from .job_worker import JobWorker
from .jobs import Job, JobStore
from .limits import Limits
from .media import MediaProbe
from .stats import Stats, TypeUsage, UserStats
from .storage import Storage
from .thumbnails import Thumbnail, ThumbnailGenerator


class Application:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage: Storage,
        file_storage: FileStorage,
        thumbnail_generator: ThumbnailGenerator,
        configuration: Configuration,
        jobs: JobStore,
        users: UserStore,
        auth: AuthConfiguration,
        limits: Limits | None = None,
        media_probe: MediaProbe | None = None,
        activity: ActivityRecorder | None = None,
    ) -> None:
        self.embedding_provider: EmbeddingProvider = embedding_provider
        self.storage: Storage = storage
        self.file_storage: FileStorage = file_storage
        self.jobs: JobStore = jobs
        self.limits = limits or Limits()
        self.users: UserStore = users
        self._admin_token = auth.admin_token.get_secret_value()
        # Places taken by uploads that are still being read, they are not in the job store yet
        self._reserved = 0
        self._queue_lock = Lock()
        self.thumbnail_generator: ThumbnailGenerator = thumbnail_generator
        self.document_processor = DocumentProcessor(
            file_storage,
            embedding_provider,
            storage,
            thumbnail_generator,
            configuration.save_strategy,
            self.limits,
            media_probe,
        )
        # Without a store the recorder is off: events are dropped on the spot
        self.activity = activity or ActivityRecorder(None, ActivityConfiguration())
        self.worker = JobWorker(
            jobs, self.document_processor, file_storage, storage, self.activity
        )

    # Starts the queue worker, jobs left over from the last run continue
    def start(self) -> None:
        self.activity.start()
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()
        self.activity.stop()

    def save(self, msg: SaveMessage) -> FailureReport | None:
        reports = self.document_processor.save(iter([msg]))
        if len(reports) == 0:
            return None
        return reports[0]

    def get_file(self, id: uuid.UUID) -> StoredFile:
        return self.file_storage.get(id)

    # JPEG, or animated WebP for an animated GIF. Audio never has its own thumbnail and files
    # stored before thumbnails appeared may lack one, both get a placeholder of their type
    def get_thumbnail(self, id: uuid.UUID) -> Thumbnail:
        descriptor = self.file_storage.descriptor(id)
        try:
            return self.file_storage.get_thumbnail(id)
        except FileNotFoundError:
            return self.thumbnail_generator.placeholder(descriptor.content_type)

    def get_by_embeddings(
        self, embeddings: list[Embeddings], n_records: int
    ) -> list[list[FileDescriptor]]:
        if n_records <= 0:
            raise NegativeRecordCount()
        if not all(
            self.embedding_provider.validate_embedding_dimension(x) for x in embeddings
        ):
            raise WrongEmbeddingsError()

        records = self.storage.get(embeddings, n_records)

        query_divided_ids = [[r.id for r in subrecords] for subrecords in records]

        flattened = [r.id for subrecord in records for r in subrecord]

        descriptors = self.file_storage.descriptors(flattened)
        descriptor_map = {x.id: x for x in descriptors}

        files = [
            [descriptor_map[x] for x in subrecord] for subrecord in query_divided_ids
        ]

        return files

    def get_by_queries(
        self, queries: list[str], n_records: int
    ) -> list[list[FileDescriptor]]:
        if n_records <= 0:
            raise NegativeRecordCount()
        embeddings = self.embedding_provider.query(queries)
        return self.get_by_embeddings(embeddings, n_records)

    # List of items that couldn't be saved
    def save_multiple(self, msgs: Iterator[SaveMessage]) -> list[FailureReport]:
        return self.document_processor.save(msgs)

    # None for a token nobody has: unknown, revoked, or of a deleted user
    def authenticate(self, token: str | None) -> Principal | None:
        if not token:
            return None
        if hmac.compare_digest(token.encode(), self._admin_token.encode()):
            return Principal(
                user_id=None, name="admin", is_admin=True, allow_embeddings=True
            )

        user = self.users.find_by_token(hash_token(token))
        if user is None:
            return None
        return Principal(
            user_id=user.id, name=user.name, allow_embeddings=user.allow_embeddings
        )

    # Ready embeddings go into the index as they are, so only users the admin trusted may send them
    def check_embeddings(self, principal: Principal) -> None:
        if not principal.allow_embeddings:
            raise EmbeddingsForbiddenError("this user may not upload ready embeddings")

    # --- admin

    # The token is returned only here and by rotate_token, the store keeps its hash
    def create_user(self, name: str, allow_embeddings: bool) -> tuple[User, str]:
        name = name.strip()
        if not USER_NAME.fullmatch(name):
            raise ValueError(
                "name must have 1-64 letters, digits, spaces, dots, dashes"
            )

        user = User(
            id=uuid.uuid7(),
            name=name,
            allow_embeddings=allow_embeddings,
            created_at=time.time(),
        )
        token = new_token()
        self.users.add(user, hash_token(token))
        return user, token

    def list_users(self) -> list[User]:
        return self.users.list()

    def update_user(self, id: uuid.UUID, allow_embeddings: bool) -> User:
        return self.users.update(id, allow_embeddings)

    # The name of a deleted user is taken for good, so a new token is also the way to get the user
    # back: the second value says that it was deleted and is active again
    def rotate_token(self, id: uuid.UUID) -> tuple[str, bool]:
        token = new_token()
        restored = self.users.set_token(id, hash_token(token))
        return token, restored

    # Revokes the token, files and statistics of the user stay
    def delete_user(self, id: uuid.UUID) -> None:
        self.users.delete(id)

    def stats(self) -> Stats:
        by_owner: dict[uuid.UUID | None, UserStats] = {}
        for u in self.users.list():
            by_owner[u.id] = UserStats(
                user_id=u.id, name=u.name, deleted=u.deleted_at is not None
            )

        for row in self.file_storage.usage():
            entry = by_owner.setdefault(
                row.owner_id,
                UserStats(
                    user_id=row.owner_id,
                    name=None if row.owner_id is None else "(unknown)",
                ),
            )
            entry.files += row.files
            entry.bytes += row.bytes
            entry.by_type[row.kind] = TypeUsage(files=row.files, bytes=row.bytes)
            if row.last_upload is not None:
                entry.last_upload = max(entry.last_upload or 0, row.last_upload)

        users = list(by_owner.values())
        return Stats(
            users=users,
            files=sum(u.files for u in users),
            bytes=sum(u.bytes for u in users),
        )

    def _activity_store(self) -> ActivityStore:
        if self.activity.store is None:
            raise ActivityDisabledError()
        return self.activity.store

    def activity_summary(self, since: float, until: float) -> ActivitySummary:
        summary = self._activity_store().summary(since, until)
        summary.jobs = self.jobs.counts()
        summary.dropped = self.activity.dropped
        return summary

    def activity_events(
        self,
        since: float,
        until: float,
        user_id: uuid.UUID | None,
        action: str | None,
        outcome: str | None,
        limit: int,
        offset: int,
    ) -> list[Event]:
        return self._activity_store().events(
            since, until, user_id, action, outcome, limit, offset
        )

    def list_files(
        self, owner_id: uuid.UUID | None, limit: int, offset: int
    ) -> list[FileRecord]:
        return self.file_storage.list_files(owner_id, limit, offset)

    # Raises FileNotFoundError for an unknown id
    def delete_file(self, id: uuid.UUID) -> None:
        self.file_storage.descriptor(id)
        self.file_storage.remove(id)
        self.storage.remove([id])

    # Takes one place in the queue for a file that is about to be read, False when it is full.
    # Looking at the count and taking the place happen under one lock, so uploads
    # running at the same time can not go over max_queue
    def _reserve(self) -> bool:
        with self._queue_lock:
            limit = self.limits.max_queue
            if limit is not None and self.jobs.active_count() + self._reserved >= limit:
                return False
            self._reserved += 1
            return True

    def _release(self, count: int) -> None:
        with self._queue_lock:
            self._reserved -= count

    # Stores the files and queues them for indexing, the caller gets job ids at once.
    # A job id is the id of its file. Ready embeddings are queued with the file,
    # checking the key for them is up to the caller
    def submit_multiple(
        self, msgs: Iterator[SaveMessage]
    ) -> tuple[list[Job], list[FailureReport]]:
        taken = 0

        def reserve() -> bool:
            nonlocal taken
            ok = self._reserve()
            taken += ok
            return ok

        try:
            return self._submit(msgs, reserve)
        finally:
            self._release(taken)

    def _submit(
        self, msgs: Iterator[SaveMessage], reserve: Callable[[], bool] | None
    ) -> tuple[list[Job], list[FailureReport]]:
        descriptors, messages, fails = self.document_processor.accept(msgs, reserve)
        jobs = [
            Job(
                id=d.id,
                original_name=d.original_name,
                description=m.text_description,
                owner_id=m.owner_id,
                embeddings=raw(m.embeddings),
                text_embeddings=raw(m.text_embeddings),
            )
            for d, m in zip(descriptors, messages)
        ]
        if jobs:
            self.jobs.add(jobs)
            self.worker.wake()
        return jobs, fails

    # Raises QueueFullError before the file is read when the queue has no room
    def submit(self, msg: SaveMessage) -> Job | FailureReport:
        if not self._reserve():
            raise QueueFullError()
        try:
            jobs, fails = self._submit(iter([msg]), None)
        finally:
            self._release(1)
        return jobs[0] if jobs else fails[0]

    def get_job(self, id: uuid.UUID) -> Job | None:
        return self.jobs.get(id)


class NegativeRecordCount(Exception):
    pass


class QueueFullError(Exception):
    pass


class EmbeddingsForbiddenError(Exception):
    pass


class ActivityDisabledError(Exception):
    pass


def raw(embeddings: Embeddings | None) -> bytes | None:
    return None if embeddings is None else embeddings.astype(np.float32).tobytes()
