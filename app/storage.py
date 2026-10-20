import uuid
from abc import ABC, abstractmethod

from .embeddings import Embeddings


class Record:
    def __init__(
        self,
        id: uuid.UUID,
        embedding: Embeddings | None,
        text_description: str | None,
        text_embedding: Embeddings | None,
    ) -> None:

        self.id: uuid.UUID = id
        self.embedding: Embeddings | None = embedding
        self.text_description: str | None = text_description
        self.text_embedding: Embeddings | None = text_embedding


class RecordNotFoundError(Exception):
    def __init__(self, ids: list[uuid.UUID]) -> None:
        self.target_id: list[uuid.UUID] = ids


class Storage(ABC):
    @abstractmethod
    def save(self, records: list[Record]) -> None:
        pass

    @abstractmethod
    def get_by_id(self, ids: list[uuid.UUID]) -> list[Record]:
        pass

    @abstractmethod
    def get(self, embeddings: list[Embeddings], n_records: int) -> list[list[Record]]:
        pass

    @abstractmethod
    def update(self, records: list[Record]):
        pass

    @abstractmethod
    def remove(self, records: list[uuid.UUID] | list[Record]):
        pass
