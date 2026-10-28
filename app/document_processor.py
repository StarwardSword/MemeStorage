import logging
import uuid
from collections.abc import Callable
from enum import Enum
from mimetypes import guess_type
from pathlib import Path
from typing import Iterator, cast

from pydantic import BaseModel, ConfigDict

from .embeddings import (
    EmbeddingProvider,
    Embeddings,
)
from .file_storage import FileStorage
from .file_storage_models import FileDescriptor, StoredFile
from .limits import LimitExceededError, Limits, limited
from .media import MediaProbe
from .storage import Record, Storage
from .thumbnails import ThumbnailGenerator

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(
        self,
        file_storage: FileStorage,
        embedding_provider: EmbeddingProvider,
        storage: Storage,
        thumbnail_generator: ThumbnailGenerator,
        save_strategy: str,
        limits: Limits | None = None,
        media_probe: MediaProbe | None = None,
    ) -> None:
        self.file_storage = file_storage
        self.embedding_provider = embedding_provider
        self.storage = storage
        self.limits = limits or Limits()
        self.media_probe = media_probe
        self.thumbnail_generator = thumbnail_generator

        if save_strategy == "safe":
            self._embeddings = self._safe_batch_embedding
            self._storage_save = self._safe_storage_save
        elif save_strategy == "fast":
            self._embeddings = self._fast_batch_embedding
            self._storage_save = self._fast_storage_save
        else:
            raise ValueError("Unexpected save strategy")

    def _fast_storage_save(self, items: list[Record]) -> list[FailureReport]:
        try:
            self.storage.save(items)

        except KeyboardInterrupt:
            raise

        except Exception:
            return [
                FailureReport(
                    item_id=x.id,
                    value="batch failure",
                    kind=FailureKind.VECTOR_STORAGE,
                )
                for x in items
            ]

        return []

    def _safe_storage_save(self, items: list[Record]) -> list[FailureReport]:
        failed: list[Record] = []

        try:
            self.storage.save(items)

        except KeyboardInterrupt:
            raise

        except Exception:
            for record in items:
                try:
                    self.storage.save([record])
                except Exception:
                    failed.append(record)

        return [
            FailureReport(
                item_id=x.id,
                kind=FailureKind.VECTOR_STORAGE,
                value=f"record {x.id}",
            )
            for x in failed
        ]

    # Embeds both files and descriptions: items are model inputs, ids and names belong to their files.
    # what tells the stage in failure reports: "embedding" or "description embedding"
    def _fast_batch_embedding(
        self,
        ids: list[uuid.UUID],
        items: list[FileDescriptor] | list[str],
        names: list[str],
        what: str,
    ) -> tuple[dict[uuid.UUID, Embeddings], list[FailureReport]]:
        try:
            embeddings = self.embedding_provider.document(items)
            return {x: e for (x, e) in zip(ids, embeddings)}, []

        except KeyboardInterrupt:
            raise

        except Exception:
            return {}, [
                FailureReport(
                    item_id=x,
                    value=f"batch failure on {what}",
                    kind=FailureKind.EMBEDDING_MODEL,
                )
                for x in ids
            ]

    # Returns dict of successful embeddings and list of failed
    def _safe_batch_embedding(
        self,
        ids: list[uuid.UUID],
        items: list[FileDescriptor] | list[str],
        names: list[str],
        what: str,
    ) -> tuple[dict[uuid.UUID, Embeddings], list[FailureReport]]:
        failed: list[tuple[uuid.UUID, str]] = []
        res: dict[uuid.UUID, Embeddings] = {}

        try:
            embeddings = self.embedding_provider.document(items)
            res = {x: e for (x, e) in zip(ids, embeddings)}

        except KeyboardInterrupt:
            raise

        except Exception:
            for id, item, name in zip(ids, items, names):
                try:
                    e = self.embedding_provider.document(cast(list[str], [item]))[0]
                    res[id] = e

                except KeyboardInterrupt:
                    raise

                except Exception:
                    failed.append((id, name))

        return res, [
            FailureReport(
                item_id=id,
                kind=FailureKind.EMBEDDING_MODEL,
                value=f"error on {what}: {name}",
            )
            for (id, name) in failed
        ]

    # Generator reads the file already written to the storage
    def _save_thumbnail(self, d: FileDescriptor) -> None:
        thumbnail = self.thumbnail_generator.generate(
            self.file_storage.get_full_path(d.id), d.content_type
        )
        if thumbnail is not None:
            self.file_storage.save_thumbnail(d.id, thumbnail)

    # reserve is asked for a place before each file is read and says no when there is none,
    # such a file and the ones after it are refused
    def _save_to_file_storage(
        self, msgs: Iterator[SaveMessage], reserve: Callable[[], bool] | None = None
    ) -> tuple[list[FileDescriptor], list[SaveMessage], list[FailureReport]]:
        res: list[FileDescriptor] = []
        fails: list[FailureReport] = []
        messages: list[SaveMessage] = []

        for number, msg in enumerate(msgs, start=1):
            id = uuid.uuid7()
            original_name = Path(msg.original_name).name

            refusal = None
            if (
                self.limits.max_batch_files is not None
                and number > self.limits.max_batch_files
            ):
                refusal = f"batch is limited to {self.limits.max_batch_files} files"
            elif reserve is not None and not reserve():
                refusal = "queue is full"
            if refusal is not None:
                fails.append(
                    FailureReport(
                        item_id=id,
                        kind=FailureKind.LIMIT_EXCEEDED,
                        value=f"{original_name}: {refusal}",
                    )
                )
                continue

            if msg.content_type is None:
                msg.content_type = guess_type(original_name)[0]

                if msg.content_type is None:
                    fails.append(
                        FailureReport(
                            item_id=id,
                            kind=FailureKind.FILE_STORAGE,
                            value=f"{msg.original_name}: unsupported Type: {msg.content_type}",
                        )
                    )
                    continue

            try:
                self.file_storage.save(
                    StoredFile(
                        id=id,
                        data=limited(msg.data, self.limits.max_bytes(msg.content_type)),
                        content_type=msg.content_type,
                        original_name=original_name,
                        owner_id=msg.owner_id,
                    )
                )
            except KeyboardInterrupt:
                raise
            except LimitExceededError as e:
                fails.append(
                    FailureReport(
                        item_id=id,
                        kind=FailureKind.LIMIT_EXCEEDED,
                        value=f"{original_name}: {e}",
                    )
                )
                continue
            except Exception:
                fails.append(
                    FailureReport(
                        item_id=id,
                        kind=FailureKind.FILE_STORAGE,
                        value=msg.original_name,
                    )
                )
                continue

            descriptor = FileDescriptor(
                id=id, original_name=original_name, content_type=msg.content_type
            )

            try:
                self._save_thumbnail(descriptor)
            except KeyboardInterrupt:
                raise
            except Exception:
                # The client gets only the file name, what went wrong stays in the log
                logger.warning("thumbnail of %s failed", original_name, exc_info=True)
                # Removed here and not by the caller: an async upload stops after this stage
                self.file_storage.remove(id)
                fails.append(
                    FailureReport(
                        item_id=id,
                        kind=FailureKind.THUMBNAIL,
                        value=f"error on thumbnail: {original_name}",
                    )
                )
                continue

            # Messages go along with descriptors, so a failed file must not shift them
            res.append(descriptor)
            messages.append(msg)

        return res, messages, fails

    # Duration is read from the stored file, so this runs after the file storage stage
    def _check_duration(self, descriptors: list[FileDescriptor]) -> list[FailureReport]:
        failed: list[FailureReport] = []
        if self.media_probe is None:
            return failed

        for d in descriptors:
            limit = self.limits.max_seconds(d.content_type)
            if limit is None:
                continue
            try:
                seconds = self.media_probe.duration(
                    self.file_storage.get_full_path(d.id), d.content_type
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failed.append(
                    FailureReport(
                        item_id=d.id,
                        kind=FailureKind.EMBEDDING_MODEL,
                        value=f"can not read media ({e}): {d.original_name}",
                    )
                )
                continue

            if seconds is not None and seconds > limit:
                failed.append(
                    FailureReport(
                        item_id=d.id,
                        kind=FailureKind.LIMIT_EXCEEDED,
                        value=f"{d.original_name}: lasts {seconds:.0f} s, the limit is {limit} s",
                    )
                )
        return failed

    def _save_to_vector_storage(
        self, descriptors: list[FileDescriptor], msgs: list[SaveMessage]
    ) -> list[FailureReport]:
        failed = self._check_duration(descriptors)
        rejected = {x.item_id for x in failed}
        pairs: list[tuple[FileDescriptor, SaveMessage]] = []

        # One report for a file even when both embeddings are wrong, so rollback gets its id once
        for d, msg in zip(descriptors, msgs):
            if d.id in rejected:
                continue
            wrong = [
                field
                for (field, value) in (
                    ("embeddings", msg.embeddings),
                    ("text_embeddings", msg.text_embeddings),
                )
                if value is not None
                and not self.embedding_provider.validate_embedding_dimension(value)
            ]
            if not wrong:
                pairs.append((d, msg))
                continue

            failed.append(
                FailureReport(
                    item_id=d.id,
                    kind=FailureKind.EMBEDDING_MODEL,
                    value=f"wrong embedding dimension ({', '.join(wrong)}): {msg.original_name}",
                )
            )

        message_map = {d.id: msg for (d, msg) in pairs}

        # Primary embeddings
        need_embeddings = [d for (d, msg) in pairs if msg.embeddings is None]

        embeddings, failed_embeddings = self._embeddings(
            [d.id for d in need_embeddings],
            need_embeddings,
            [d.original_name for d in need_embeddings],
            "embedding",
        )
        failed += failed_embeddings

        for id in embeddings:
            message_map[id].embeddings = embeddings[id]

        # Secondary embeddings, only for files that still have primary ones
        need_text = [
            (d, msg.text_description)
            for (d, msg) in pairs
            if msg.embeddings is not None
            and msg.text_embeddings is None
            and msg.text_description is not None
        ]

        embeddings, failed_embeddings = self._embeddings(
            [d.id for (d, text) in need_text],
            [text for (d, text) in need_text],
            [d.original_name for (d, text) in need_text],
            "description embedding",
        )
        failed += failed_embeddings

        for id in embeddings:
            message_map[id].text_embeddings = embeddings[id]

        # Vector database. A record without its description embedding would fail the whole batch there
        failed_ids = {x.item_id for x in failed}
        records = [
            Record(
                id=d.id,
                embedding=msg.embeddings,
                text_description=msg.text_description,
                text_embedding=msg.text_embeddings,
            )
            for (d, msg) in pairs
            if msg.embeddings is not None and d.id not in failed_ids
        ]

        failed_records = self._storage_save(records)
        for f in failed_records:
            f.value = message_map[f.item_id].original_name

        failed += failed_records

        return failed

    # Writes the files to disk and nothing else, indexing is a separate step so it can wait in a queue
    def accept(
        self, msgs: Iterator[SaveMessage], reserve: Callable[[], bool] | None = None
    ) -> tuple[list[FileDescriptor], list[SaveMessage], list[FailureReport]]:
        return self._save_to_file_storage(msgs, reserve)

    # Removes files of the failed reports from the disk and from the vector storage
    def rollback(self, fails: list[FailureReport]) -> None:
        # Vector storage refuses repeated ids
        ids = list(dict.fromkeys(x.item_id for x in fails))
        for id in ids:
            self.file_storage.remove(id)

        if len(ids) != 0:
            self.storage.remove(ids)

    # Embeds and indexes files that accept stored, a file that fails is removed
    def index(
        self, descriptors: list[FileDescriptor], msgs: list[SaveMessage]
    ) -> list[FailureReport]:
        fails = self._save_to_vector_storage(descriptors, msgs)
        self.rollback(fails)
        return fails

    def save(self, msgs: Iterator[SaveMessage]) -> list[FailureReport]:
        descriptors, message_list, fails = self.accept(msgs)

        fails += self._save_to_vector_storage(descriptors, message_list)
        self.rollback(fails)

        return fails


class FailureKind(Enum):
    FILE_STORAGE = "File storage"
    THUMBNAIL = "Thumbnail"
    EMBEDDING_MODEL = "Embedding model"
    VECTOR_STORAGE = "Vector storage"
    LIMIT_EXCEEDED = "Limit exceeded"


class FailureReport(BaseModel):
    kind: FailureKind
    value: str
    item_id: uuid.UUID


class SaveMessage(BaseModel):
    # Iterator and numpy array have no pydantic schema, they are checked with isinstance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    original_name: str
    data: Iterator[bytes]
    content_type: str | None = None
    embeddings: Embeddings | None = None
    text_description: str | None = None
    text_embeddings: Embeddings | None = None
    # Uploader, None for the admin
    owner_id: uuid.UUID | None = None
