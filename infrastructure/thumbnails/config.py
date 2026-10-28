from pydantic import BaseModel, Field


class Configuration(BaseModel):
    # Longest side in pixels
    size: int = Field(default=320, gt=0)
    quality: int = Field(default=85, ge=1, le=100)
