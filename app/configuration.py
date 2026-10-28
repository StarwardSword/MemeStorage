from typing import Literal

from pydantic import BaseModel, Field, SecretStr


class Configuration(BaseModel):
    save_strategy: Literal["safe", "fast"]


class AuthConfiguration(BaseModel):
    # Opens the admin API and everything else. Users get their own tokens from it
    admin_token: SecretStr = Field(min_length=24)


class ActivityConfiguration(BaseModel):
    """Log of everything that happens: who did what, how it ended, how long it took"""

    enabled: bool = True
    # Older events are deleted
    retention_days: int = Field(default=90, ge=1)
    # Text of search queries goes to the log and to the top of queries. Turn off to keep searches private
    record_queries: bool = True
