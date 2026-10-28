from collections.abc import Iterator

from pydantic import BaseModel, Field, field_validator

from .media import is_gif

MB = 1024 * 1024


class LimitExceededError(Exception):
    pass


class Limits(BaseModel):
    """Every limit is optional, a missing or zero value means no limit"""

    max_image_mb: int | None = Field(default=None, ge=1)
    max_video_mb: int | None = Field(default=None, ge=1)
    max_audio_mb: int | None = Field(default=None, ge=1)
    max_video_seconds: int | None = Field(default=None, ge=1)
    max_audio_seconds: int | None = Field(default=None, ge=1)
    # A GIF is an image by its size (max_image_mb), by its length and frames it is like a video
    max_gif_seconds: int | None = Field(default=None, ge=1)
    max_gif_frames: int | None = Field(default=None, ge=1)
    # Files in one batch request, the rest of them is reported as failed
    max_batch_files: int | None = Field(default=None, ge=1)
    # Jobs waiting or running at the same time, async uploads over it are refused
    max_queue: int | None = Field(default=None, ge=1)

    @field_validator("*", mode="before")
    @classmethod
    def zero_is_no_limit(cls, v: object) -> object:
        return None if v == 0 else v

    def max_bytes(self, content_type: str | None) -> int | None:
        mb = {
            "image": self.max_image_mb,
            "video": self.max_video_mb,
            "audio": self.max_audio_mb,
        }.get((content_type or "").split("/")[0])
        return None if mb is None else mb * MB

    def max_seconds(self, content_type: str | None) -> int | None:
        if is_gif(content_type):
            return self.max_gif_seconds
        return {
            "video": self.max_video_seconds,
            "audio": self.max_audio_seconds,
        }.get((content_type or "").split("/")[0])

    def max_frames(self, content_type: str | None) -> int | None:
        return self.max_gif_frames if is_gif(content_type) else None


def limited(data: Iterator[bytes], max_bytes: int | None) -> Iterator[bytes]:
    """Passes the chunks through and stops with LimitExceededError once they add up to too much"""
    if max_bytes is None:
        yield from data
        return

    total = 0
    for chunk in data:
        total += len(chunk)
        if total > max_bytes:
            raise LimitExceededError(f"file is larger than {max_bytes // MB} MB")
        yield chunk
