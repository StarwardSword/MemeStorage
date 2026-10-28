import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from .file_storage_models import FileDescriptor, StoredFile


class FileStorage(ABC):
    @abstractmethod
    def save(self, file: StoredFile) -> Path:
        """
        Reads and closes specified byte stream in the StoredFile
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, id: uuid.UUID) -> StoredFile:
        """
        Returns StoredFile object with opened byte stream

        It is up to caller to close the stream
        """
        raise NotImplementedError

    @abstractmethod
    def descriptor(self, id: uuid.UUID) -> FileDescriptor:
        """
        Returns grouped information about the file
        """
        raise NotImplementedError

    @abstractmethod
    def descriptors(self, ids: list[uuid.UUID]) -> list[FileDescriptor]:
        """
        Returns a list of file descriptors ordered by input ids
        """
        raise NotImplementedError

    @abstractmethod
    def remove(self, id: uuid.UUID) -> None:
        """
        Removes the file together with its thumbnail
        """
        raise NotImplementedError

    @abstractmethod
    def save_thumbnail(self, id: uuid.UUID, data: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_thumbnail(self, id: uuid.UUID) -> bytes:
        """
        Raises FileNotFoundError when the file has no thumbnail
        """
        raise NotImplementedError

    @abstractmethod
    def get_full_path(self, id: uuid.UUID) -> Path:
        raise NotImplementedError
