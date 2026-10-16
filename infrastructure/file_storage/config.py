from pydantic import BaseModel, field_validator

class Configuration(BaseModel):
    root: str
    sqlite_connection_string: str
    chunk_size: int = 64 * 1024