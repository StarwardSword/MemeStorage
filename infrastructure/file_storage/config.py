from pydantic import BaseModel


class Configuration(BaseModel):
    root: str
    thumbnails_root: str
    sqlite_connection_string: str
    chunk_size: int = 64 * 1024
