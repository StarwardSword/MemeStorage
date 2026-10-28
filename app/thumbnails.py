from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

THUMBNAIL_CONTENT_TYPE = "image/jpeg"
# An animated GIF gets a thumbnail that moves
ANIMATED_THUMBNAIL_CONTENT_TYPE = "image/webp"


class Thumbnail(BaseModel):
    data: bytes
    content_type: str


class ThumbnailGenerator(ABC):
    @abstractmethod
    def generate(self, source: Path, content_type: str) -> Thumbnail | None:
        """
        Returns the thumbnail of the file, None when files of this type have no own thumbnail.
        JPEG, or animated WebP for an animated GIF

        Raises when the file can not be decoded
        """
        raise NotImplementedError

    @abstractmethod
    def placeholder(self, content_type: str) -> Thumbnail:
        """
        Returns the JPEG shown instead of the thumbnail when the file has none
        """
        raise NotImplementedError
