from abc import ABC, abstractmethod
from pathlib import Path

GIF = "image/gif"


def is_gif(content_type: str | None) -> bool:
    return content_type == GIF


class MediaProbe(ABC):
    @abstractmethod
    def duration(self, path: Path, content_type: str) -> float | None:
        """Length in seconds, None when the container does not tell it. Raises when the file can not be read"""
        raise NotImplementedError

    def frames(self, path: Path, content_type: str) -> int | None:
        """Number of frames of a GIF, None for other files. Raises when the file can not be read"""
        return None
