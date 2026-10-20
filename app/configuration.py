from typing import Literal

from pydantic import BaseModel


class Configuration(BaseModel):
    save_strategy: Literal["safe", "fast"]
