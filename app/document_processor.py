import uuid
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
from .storage import Record, Storage
from .thumbnails import ThumbnailGenerator


class DocumentProcessor:
    def __init__(
        self,
        file_storage: FileStorage,
        embedding_provider: EmbeddingProvider,
        storage: Storage,
        thumbnail_generator: ThumbnailGenerator,
        save_strategy: str,
    ) -> None:
        self.file_storage = file_storage
        self.embedding_provider = embedding_provider
        self.storage = storage
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
                    value=f"batch failure",
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
                except:
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

    def _save_to_file_storage(
        self, msgs: Iterator[SaveMessage]
    ) -> tuple[list[FileDescriptor], list[SaveMessage], list[FailureReport]]:
        res: list[FileDescriptor] = []
        fails: list[FailureReport] = []
        messages: list[SaveMessage] = []

        for msg in msgs:
            id = uuid.uuid7()
            original_name = Path(msg.original_name).name

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
                        data=msg.data,
                        content_type=msg.content_type,
                        original_name=original_name,
                    )
                )
            except KeyboardInterrupt:
                raise
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

    def _save_to_vector_storage(
        self, descriptors: list[FileDescriptor], msgs: list[SaveMessage]
    ) -> list[FailureReport]:
        failed: list[FailureReport] = []
        pairs: list[tuple[FileDescriptor, SaveMessage]] = []

        # One report for a file even when both embeddings are wrong, so rollback gets its id once
        for d, msg in zip(descriptors, msgs):
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

    def save(self, msgs: Iterator[SaveMessage]) -> list[FailureReport]:
        descriptors, message_list, fails = self._save_to_file_storage(msgs)

        fails += self._save_to_vector_storage(descriptors, message_list)

        # Vector storage refuses repeated ids
        ids = list(dict.fromkeys(x.item_id for x in fails))
        for id in ids:
            self.file_storage.remove(id)

        if len(ids) != 0:
            self.storage.remove(ids)

        return fails


class FailureKind(Enum):
    FILE_STORAGE = "File storage"
    THUMBNAIL = "Thumbnail"
    EMBEDDING_MODEL = "Embedding model"
    VECTOR_STORAGE = "Vector storage"


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
