from pydantic import BaseModel, field_validator
from torch.cuda import is_available as cuda_available


class Configuration(BaseModel):
    device: str = "cpu"
    dtype: str = "float32"
    semaphore_count: int = 4
    max_batch: int

    @field_validator("device")
    @classmethod
    def device_validator(cls, v: str) -> str:
        allowed_device = frozenset(["cpu", "cuda"])
        if v not in allowed_device:
            raise ValueError(
                f'unexpected device "{v}"; expected=[{", ".join(allowed_device)}]'
            )

        if v == "cuda" and not cuda_available():
            raise ValueError(f"cuda selected but torch.cuda is not available")

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
