import uuid
from typing import Iterator


class StoredFile:
    """
    File data and opened byte stream

    It is up to receiver to close the stream
    """

    def __init__(
        self,
        id: uuid.UUID,
        data: Iterator[bytes],
        content_type: str,
        original_name: str,
        size: int | None = None,
    ):
        self.id: uuid.UUID = id
        self.data: Iterator[bytes] = data
        self.content_type: str = content_type
        self.original_name: str = original_name
        self.size: int | None = size


class FileDescriptor:
    """
    File data without byte stream
    """

    def __init__(self, id: uuid.UUID, original_name: str, content_type: str):
        self.id: uuid.UUID = id
        self.original_name: str = original_name
        self.content_type: str = content_type
