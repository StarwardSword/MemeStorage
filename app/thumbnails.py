from abc import ABC, abstractmethod
from pathlib import Path

THUMBNAIL_CONTENT_TYPE = "image/jpeg"


class ThumbnailGenerator(ABC):
    @abstractmethod
    def generate(self, source: Path, content_type: str) -> bytes | None:
        """
        Returns JPEG thumbnail of the file, None when files of this type have no own thumbnail

        Raises when the file can not be decoded
        """
        raise NotImplementedError
