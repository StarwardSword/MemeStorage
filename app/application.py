import uuid
from pathlib import Path
from typing import Iterator
from mimetypes import guess_type
from .embeddings import EmbeddingProvider, Embeddings
from .storage import Storage, Record
from .file_storage import FileStorage
from .file_storage_models import StoredFile, FileDescriptor

class Application:
    def __init__(self, embedding_provider: EmbeddingProvider, storage: Storage, file_storage: FileStorage) -> None:
        self.embedding_provider: EmbeddingProvider = embedding_provider
        self.storage: Storage = storage
        self.file_storage: FileStorage = file_storage

    def _save_to_file_storage(self, id: uuid.UUID, data: Iterator[bytes], original_name: str, content_type: str | None) -> FileDescriptor:
            original_name = Path(original_name).name
            if content_type is None:
                content_type = guess_type(original_name)[0]
                
                if content_type is None:
                    raise ValueError(content_type)

            self.file_storage.save(StoredFile(
                id=id,
                data=data,
                content_type=content_type,
                original_name=original_name
            ))

            return FileDescriptor(
                 id=id,
                 original_name=original_name,
                 content_type=content_type,
            )

    def _save_to_vector_storage(self, descriptor: FileDescriptor, embeddings: Embeddings | None, text_description: str | None, text_embeddings: Embeddings | None) -> Record:
            if embeddings is None:
                embeddings = self.embedding_provider.document([descriptor])[0]

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

    def save(self
            , original_name: str
            , data: Iterator[bytes]
            , content_type: str | None = None
            , embeddings: Embeddings | None = None
            , text_description: str | None = None
            , text_embeddings: Embeddings | None = None
        ) -> uuid.UUID:

            id = uuid.uuid7()
            
            descriptor = self._save_to_file_storage(id, data, original_name, content_type)

            try:
                record = self._save_to_vector_storage(descriptor, embeddings, text_description, text_embeddings)
            except:
                self.file_storage.remove(id)
                raise

            return id

    def get_file(self, id: uuid.UUID) -> StoredFile:
        return self.file_storage.get(id)

    def get_by_embeddings(self, embeddings: list[Embeddings], n_records: int) -> list[list[FileDescriptor]]:
        records = self.storage.get(embeddings, n_records)

        query_divided_ids = [[
                r.id for r in subrecords
            ] for subrecords in records
        ]

        flattened = [
            r.id
            for subrecord in records
            for r in subrecord
        ]

        descriptors = self.file_storage.descriptors(flattened)
        descriptor_map = {x.id: x for x in descriptors}

        files = [[
                descriptor_map[x] for x in subrecord
            ] for subrecord in query_divided_ids
        ]

        return files
 
    def get_by_queries(self, queries: list[str], n_records: int) -> list[list[FileDescriptor]]:
        embeddings = self.embedding_provider.query(queries)
        return self.get_by_embeddings(embeddings, n_records)