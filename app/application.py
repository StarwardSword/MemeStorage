import uuid
from pathlib import Path
from typing import Iterator
from mimetypes import guess_type
from .embeddings import (
    EmbeddingProvider,
    Embeddings,
    WrongEmbeddingsError,
    UnsupportedTypeError,
)
from .storage import Storage, Record
from .file_storage import FileStorage
from .file_storage_models import StoredFile, FileDescriptor
from pydantic import BaseModel, ConfigDict


class Application:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage: Storage,
        file_storage: FileStorage,
    ) -> None:
        self.embedding_provider: EmbeddingProvider = embedding_provider
        self.storage: Storage = storage
        self.file_storage: FileStorage = file_storage

    def _save_to_file_storage(
        self,
        id: uuid.UUID,
        data: Iterator[bytes],
        original_name: str,
        content_type: str | None,
    ) -> FileDescriptor:
        original_name = Path(original_name).name
        if content_type is None:
            content_type = guess_type(original_name)[0]

            # None means the type is unknown, not that it is unsupported
            if content_type is None:
                raise UnsupportedTypeError(None)

        self.file_storage.save(
            StoredFile(
                id=id, data=data, content_type=content_type, original_name=original_name
            )
        )

        return FileDescriptor(
            id=id,
            original_name=original_name,
            content_type=content_type,
        )

    def _save_to_vector_storage(
        self,
        descriptor: FileDescriptor,
        embeddings: Embeddings | None,
        text_description: str | None,
        text_embeddings: Embeddings | None,
    ) -> Record:
        if embeddings is None:
            embeddings = self.embedding_provider.document([descriptor])[0]
        else:
            if not self.embedding_provider.validate_embedding_dimension(embeddings):
                raise WrongEmbeddingsError()

        if text_description is not None:
            text_embeddings = self.embedding_provider.document([text_description])[0]

        record = Record(
            id=descriptor.id,
            embedding=embeddings,
            text_description=text_description,
            text_embedding=text_embeddings,
        )

        self.storage.save([record])
        return record

    def save(self, msg: SaveMessage) -> uuid.UUID:

        id = uuid.uuid7()

        descriptor = self._save_to_file_storage(
            id, msg.data, msg.original_name, msg.content_type
        )

        try:
            record = self._save_to_vector_storage(
                descriptor, msg.embeddings, msg.text_description, msg.text_embeddings
            )
        except:
            self.file_storage.remove(id)
            self.storage.remove([id])
            raise

        return id

    def get_file(self, id: uuid.UUID) -> StoredFile:
        return self.file_storage.get(id)

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

    # List of saved ids and list of items that couldn't be saved
    def save_multiple(self, msgs: Iterator[SaveMessage]) -> list[str]:
        errored: list[str] = []
        descriptors: list[FileDescriptor] = []

        for msg in msgs:
            id = uuid.uuid7()
            try:
                descriptor = self._save_to_file_storage(
                    id, msg.data, msg.original_name, msg.content_type
                )
                descriptors.append(descriptor)
            except Exception:
                # todo: log
                self.file_storage.remove(id)
                errored.append(msg.original_name)

        embeddings = self.embedding_provider.document(descriptors)
        records = [
            Record(
                id=x.id,
                embedding=embeddings[i],
                text_description=None,
                text_embedding=None,
            )
            for (i, x) in enumerate(descriptors)
        ]

        self.storage.save(records)
        return errored


class NegativeRecordCount(Exception):
    pass


class SaveMessage(BaseModel):
    # Iterator and numpy array have no pydantic schema, they are checked with isinstance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    original_name: str
    data: Iterator[bytes]
    content_type: str | None = None
    embeddings: Embeddings | None = None
    text_description: str | None = None
    text_embeddings: Embeddings | None = None
