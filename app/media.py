from abc import ABC, abstractmethod
from pathlib import Path


class MediaProbe(ABC):
    @abstractmethod
    def duration(self, path: Path, content_type: str) -> float | None:
        """Length in seconds, None when the container does not tell it. Raises when the file can not be read"""
        raise NotImplementedError
