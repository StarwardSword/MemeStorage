from typing import Literal

from pydantic import BaseModel, Field, field_validator
from torch.cuda import is_available as cuda_available


class Configuration(BaseModel):
    # torch runs the original weights, onnx runs int8 ones: less memory, same vectors up to ~0.002
    backend: Literal["torch", "onnx"] = "torch"
    device: str = "cpu"
    dtype: str = "float32"
    semaphore_count: int = Field(default=4, ge=1)
    max_batch: int = Field(ge=1)
    threads: int | None = Field(default=None, ge=1)
    truncate_dim: Literal[128, 256, 512, 768] | None = None
    idle_unload_seconds: float | None = Field(default=None, gt=0)
    # How many search texts keep their embedding in memory, 0 turns the cache off
    query_cache_size: int = Field(default=256, ge=0)

    @field_validator("device")
    @classmethod
    def device_validator(cls, v: str) -> str:
        allowed_device = frozenset(["cpu", "cuda"])
        if v not in allowed_device:
            raise ValueError(
                f'unexpected device "{v}"; expected=[{", ".join(allowed_device)}]'
            )

        if v == "cuda" and not cuda_available():
            raise ValueError("cuda selected but torch.cuda is not available")

        return v

    @field_validator("dtype")
    @classmethod
    def dtype_validator(cls, v: str) -> str:
        v = v.removeprefix("torch.")
        allowed_types = frozenset(["float16", "float32", "bfloat16"])
        if v not in allowed_types:
            raise ValueError(
                f'unexpected type "{v}"; expected=[{", ".join(allowed_types)}]'
            )

        return v
