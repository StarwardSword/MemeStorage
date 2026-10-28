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

from app import User

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


SaveFailureKind = Literal[
    "unsupported_type",
    "file_storage",
    "thumbnail",
    "embedding_model",
    "vector_storage",
    "limit_exceeded",
]
JobStatus = Literal["pending", "running", "done", "failed"]


class SaveFailure(BaseModel):
    kind: SaveFailureKind = Field(
        description="Stage that failed. `unsupported_type`: the file is not an image, video or audio "
        "and was not passed to the application; `thumbnail`: the file could not be decoded to build its thumbnail; "
        "`embedding_model` includes a wrong embedding dimension; "
        "`limit_exceeded`: the file is too big or too long, or the batch or the queue is full"
    )
    message: str = Field(
        description="Usually holds the file name, but not always: "
        'with save strategy "fast" a failed batch is reported without file names'
    )


class SaveMultipleResponse(BaseModel):
    failed: list[SaveFailure] = Field(
        description="Files that were not saved, the rest are saved"
    )


class JobAccepted(BaseModel):
    id: uuid.UUID = Field(
        description="Job id, it is also the id of the stored file: `GET /jobs/{id}` tells when it is searchable"
    )


class SubmittedFile(JobAccepted):
    filename: str


class SubmitMultipleResponse(BaseModel):
    jobs: list[SubmittedFile] = Field(description="Files that were queued")
    failed: list[SaveFailure] = Field(
        description="Files that were refused, the rest are queued"
    )


class JobInfo(BaseModel):
    id: uuid.UUID
    original_name: str
    status: JobStatus = Field(
        description="`pending` waits in the queue, `running` is being indexed, "
        "`done` is searchable, `failed` was removed and `error` tells why"
    )
    error: SaveFailure | None = None


class UserCreate(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=64,
        description="Letters, digits, spaces, dots and dashes. Unique, a deleted user keeps its name",
    )
    allow_embeddings: bool = Field(
        default=False,
        description="May upload ready embeddings. They go into the index as they are, so give it to trusted users",
    )


class UserUpdate(BaseModel):
    allow_embeddings: bool


class UserCreated(BaseModel):
    user: User
    token: str = Field(
        description="Token of the user for `Authorization: Bearer`. Shown only now"
    )


class TokenResponse(BaseModel):
    token: str = Field(description="New token, the old one does not work anymore")
    restored: bool = Field(
        description="The user was deleted and is active again with this token"
    )


class Error(BaseModel):
    detail: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"
