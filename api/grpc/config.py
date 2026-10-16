from pydantic import BaseModel

class Configuration(BaseModel):
    max_workers: int = 10
    port: str