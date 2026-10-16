import uuid
from typing import Annotated, Literal
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    Json,
    TypeAdapter,
    ValidationError,
    WithJsonSchema,
    field_validator,
)

MAX_BATCH = 32
MAX_COUNT = 100
DEFAULT_COUNT = 10

Vector = Annotated[list[float], Field(min_length=1)]
Count = Annotated[
    int,
    Field(ge=1, le=MAX_COUNT, description="Number of files to return for each query"),
]

_vector = TypeAdapter(Vector)


def _parse_vector(value: str) -> list[float]:
    try:
        return _vector.validate_json(value)
    except ValidationError as e:
        raise ValueError("; ".join(x["msg"] for x in e.errors())) from None


# Form field holding a JSON array, validated value is list[float].
# Declared as str because FastAPI collects repeated form fields into list-typed parameters instead of parsing JSON
JsonVector = Annotated[
    str,
    AfterValidator(_parse_vector),
    WithJsonSchema(TypeAdapter(Json[Vector]).json_schema()),
]


class FileInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_name: str
    content_type: str


class FileList(BaseModel):
    items: list[FileInfo] = Field(description="Closest files first")


class SearchResponse(BaseModel):
    results: list[FileList] = Field(
        description="One list for each query, in request order"
    )


class QuerySearchRequest(BaseModel):
    queries: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, max_length=MAX_BATCH, description="Text queries"
    )
    count: Count = DEFAULT_COUNT


class EmbeddingSearchRequest(BaseModel):
    embeddings: list[Vector] = Field(
        min_length=1,
        max_length=MAX_BATCH,
        description="Query embeddings of the same dimension as stored ones",
    )
    count: Count = DEFAULT_COUNT

    @field_validator("embeddings")
    @classmethod
    def same_dimension(cls, v: list[list[float]]) -> list[list[float]]:
        if len({len(x) for x in v}) > 1:
            raise ValueError("all embeddings must have the same dimension")
        return v


class SaveResponse(BaseModel):
    id: uuid.UUID


class Error(BaseModel):
    detail: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"
