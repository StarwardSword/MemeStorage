import uuid

from pydantic import BaseModel


class TypeUsage(BaseModel):
    files: int = 0
    bytes: int = 0


class UserStats(BaseModel):
    # None for files with no owner: uploaded by the admin or before users existed
    user_id: uuid.UUID | None
    name: str | None
    deleted: bool = False
    files: int = 0
    bytes: int = 0
    # Keyed by image, video and audio
    by_type: dict[str, TypeUsage] = {}
    last_upload: float | None = None


class Stats(BaseModel):
    users: list[UserStats]
    files: int
    bytes: int
