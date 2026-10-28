import uuid
from typing import Iterator

from pydantic import BaseModel


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
        owner_id: uuid.UUID | None = None,
    ):
        self.id: uuid.UUID = id
        self.data: Iterator[bytes] = data
        self.content_type: str = content_type
        self.original_name: str = original_name
        self.size: int | None = size
        self.owner_id: uuid.UUID | None = owner_id


class FileDescriptor:
    """
    File data without byte stream
    """

    def __init__(self, id: uuid.UUID, original_name: str, content_type: str):
        self.id: uuid.UUID = id
        self.original_name: str = original_name
        self.content_type: str = content_type


class FileRecord(BaseModel):
    """What the admin sees about a stored file. Files stored before owners existed have no size and time"""

    id: uuid.UUID
    original_name: str
    content_type: str
    owner_id: uuid.UUID | None = None
    size: int | None = None
    created_at: float | None = None


class OwnerUsage(BaseModel):
    """Files of one owner and one kind (image, video, audio)"""

    owner_id: uuid.UUID | None
    kind: str
    files: int
    bytes: int
    last_upload: float | None = None
