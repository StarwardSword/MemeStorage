import numpy as np
from abc import ABC, abstractmethod
from typing import overload
from numpy.typing import NDArray
from .file_storage_models import FileDescriptor

Embeddings = NDArray[np.float32]


class WrongEmbeddingsError(Exception):
    pass


class UnsupportedTypeError(Exception):
    pass


class EmbeddingProvider(ABC):
    @overload
    def document(self, items: list[FileDescriptor]) -> list[Embeddings]:
        raise NotImplementedError

    @overload
    def document(self, items: list[str]) -> list[Embeddings]:
        raise NotImplementedError

    @abstractmethod
    def document(self, items: list[FileDescriptor] | list[str]) -> list[Embeddings]:
        raise NotImplementedError

    @abstractmethod
    def query(self, queries: list[str]) -> list[Embeddings]:
        raise NotImplementedError

    @abstractmethod
    def validate_embedding_dimension(self, embedding: Embeddings) -> bool:
        raise NotImplementedError
