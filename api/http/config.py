from pydantic import BaseModel


class Configuration(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    max_workers: int = 10
