import uuid
from typing import Iterator

from .configuration import Configuration
from .document_processor import DocumentProcessor, FailureReport, SaveMessage
from .embeddings import (
    EmbeddingProvider,
    Embeddings,
    WrongEmbeddingsError,
)
from .file_storage import FileStorage
from .file_storage_models import FileDescriptor, StoredFile
from .storage import Storage


class Application:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage: Storage,
        file_storage: FileStorage,
        configuration: Configuration,
    ) -> None:
        self.embedding_provider: EmbeddingProvider = embedding_provider
        self.storage: Storage = storage
        self.file_storage: FileStorage = file_storage
        self.document_processor = DocumentProcessor(
            file_storage, embedding_provider, storage, configuration.save_strategy
        )

    def save(self, msg: SaveMessage) -> FailureReport | None:
        reports = self.document_processor.save(iter([msg]))
        if len(reports) == 0:
            return None
        return reports[0]

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

    # List of items that couldn't be saved
    def save_multiple(self, msgs: Iterator[SaveMessage]) -> list[FailureReport]:
        return self.document_processor.save(msgs)


class NegativeRecordCount(Exception):
    pass
