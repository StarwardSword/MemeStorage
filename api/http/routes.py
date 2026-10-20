import uuid
from functools import partial
from mimetypes import guess_type
from typing import Annotated, Any, Iterator, TypeIs
from urllib.parse import quote

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse

from app import (
    Application,
    FailureKind,
    FailureReport,
    FileDescriptor,
    SaveMessage,
    WrongEmbeddingsError,
)

from .schemas import (
    DEFAULT_COUNT,
    MAX_COUNT,
    EmbeddingSearchRequest,
    Error,
    FileInfo,
    FileList,
    Health,
    JsonVector,
    QuerySearchRequest,
    SaveFailure,
    SaveFailureKind,
    SaveMultipleResponse,
    SearchResponse,
)

CHUNK_SIZE = 64 * 1024
MEDIA_PREFIXES = ("image/", "video/", "audio/")
# Sent by browsers and curl when they do not know the type, server guesses it by file name then
GENERIC_CONTENT_TYPES = ("", "application/octet-stream")

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": Error, "description": "File not found"}
}

FAILURE_KINDS: dict[FailureKind, SaveFailureKind] = {
    FailureKind.FILE_STORAGE: "file_storage",
    FailureKind.EMBEDDING_MODEL: "embedding_model",
    FailureKind.VECTOR_STORAGE: "vector_storage",
}


async def get_application(request: Request) -> Application:
    return request.app.state.application


ApplicationDep = Annotated[Application, Depends(get_application)]


# FileNotFoundError message may hold a server path, so the client gets only the id
def file_not_found(file_id: uuid.UUID) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"file not found: {file_id}")


# Dimension is known only to the model, so it is checked by the application but reported like request validation
def wrong_embeddings(field: str) -> RequestValidationError:
    return RequestValidationError(
        [
            {
                "type": "value_error",
                "loc": ("body", field),
                "msg": "embedding dimension does not match the model",
            }
        ]
    )


# Client errors are checked before saving, so what the application reports is a server failure
def save_failed(report: FailureReport) -> HTTPException:
    return HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR, f"{report.kind.value}: {report.value}"
    )


def save_failure(report: FailureReport) -> SaveFailure:
    return SaveFailure(kind=FAILURE_KINDS[report.kind], message=report.value)


def detect_content_type(file: UploadFile) -> str | None:
    content_type = file.content_type
    if content_type is None or content_type in GENERIC_CONTENT_TYPES:
        content_type = guess_type(file.filename or "")[0]
    return content_type


def is_media(content_type: str | None) -> TypeIs[str]:
    return content_type is not None and content_type.startswith(MEDIA_PREFIXES)


def unsupported_type_message(content_type: str | None) -> str:
    return (
        f"unsupported content type {content_type}, expected image/*, video/* or audio/*"
    )


def media_type(file: UploadFile) -> str:
    content_type = detect_content_type(file)
    if not is_media(content_type):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            unsupported_type_message(content_type),
        )
    return content_type


# filename* carries the real name (RFC 6266), filename is an ASCII fallback for old clients
def content_disposition(filename: str, attachment: bool) -> str:
    fallback = "".join(
        c if c.isascii() and c.isprintable() and c not in '"\\' else "_"
        for c in filename
    )
    return f"{'attachment' if attachment else 'inline'}; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"


def file_data(file: UploadFile) -> Iterator[bytes]:
    return iter(partial(file.file.read, CHUNK_SIZE), b"")


def file_list(descriptors: list[FileDescriptor]) -> FileList:
    return FileList(items=[FileInfo.model_validate(x) for x in descriptors])


def search_response(found: list[list[FileDescriptor]]) -> SearchResponse:
    return SearchResponse(results=[file_list(row) for row in found])


# --- files

files = APIRouter(prefix="/files", tags=["files"])


@files.post(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Upload a file",
    description="Stores an image, video or audio file and indexes it for search. "
    "Content type is taken from the file part, or guessed by file name when it is missing or `application/octet-stream`. "
    "`embeddings` and `text_embeddings` must have the dimension of the server model, otherwise the response is `422`. "
    "Id of the saved file is not returned, the file is found by search.",
    response_description="File stored",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": Error,
            "description": "File part has no file name",
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "model": Error,
            "description": "File is not an image, video or audio",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": Error,
            "description": "File storage, embedding model or vector storage failed, nothing is stored",
        },
    },
)
def save_file(
    application: ApplicationDep,
    file: Annotated[UploadFile, File(description="Image, video or audio file")],
    description: Annotated[
        str | None, Form(description="Text description, the file is also found by it")
    ] = None,
    embeddings: Annotated[
        JsonVector | None,
        Form(
            description="Precomputed file embedding as a JSON array of numbers, computed by the server when omitted"
        ),
    ] = None,
    text_embeddings: Annotated[
        JsonVector | None,
        Form(
            description="Precomputed description embedding as a JSON array of numbers"
        ),
    ] = None,
) -> None:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file must have a name")

    msg = SaveMessage(
        original_name=file.filename,
        data=file_data(file),
        content_type=media_type(file),
        text_description=description,
        embeddings=np.asarray(embeddings, dtype=np.float32)
        if embeddings is not None
        else None,
        text_embeddings=np.asarray(text_embeddings, dtype=np.float32)
        if text_embeddings is not None
        else None,
    )

    # Application reports a wrong dimension as a model failure, here it is a client error
    for field, value in (
        ("embeddings", msg.embeddings),
        ("text_embeddings", msg.text_embeddings),
    ):
        if (
            value is not None
            and not application.embedding_provider.validate_embedding_dimension(value)
        ):
            raise wrong_embeddings(field)

    failure = application.save(msg)
    if failure is not None:
        raise save_failed(failure)


@files.post(
    "/batch",
    summary="Upload several files",
    description="Stores image, video and audio files and indexes them for search in one pass. "
    "Content type of each file is detected as in `POST /files`. "
    "A file that is not an image, video or audio or can not be stored does not fail the request, "
    "it is reported in `failed` with the failed stage. Ids of saved files are not returned. "
    "Descriptions and precomputed embeddings are not supported here.",
    response_description="Files processed, some of them may have failed",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": Error,
            "description": "A file part has no file name, nothing is stored",
        },
    },
)
def save_files(
    application: ApplicationDep,
    # Alias keeps the form field name without shadowing the router
    uploads: Annotated[
        list[UploadFile],
        File(alias="files", description="Image, video or audio files"),
    ],
) -> SaveMultipleResponse:
    if not all(x.filename for x in uploads):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "every file must have a name")

    # With save strategy "fast" the model refuses a non-media file for the whole batch,
    # so such files are not passed to the application
    rejected: list[SaveFailure] = []
    messages: list[SaveMessage] = []
    for upload in uploads:
        content_type = detect_content_type(upload)
        if not is_media(content_type):
            rejected.append(
                SaveFailure(
                    kind="unsupported_type",
                    message=f"{upload.filename}: {unsupported_type_message(content_type)}",
                )
            )
            continue
        messages.append(
            SaveMessage(
                original_name=upload.filename or "",
                data=file_data(upload),
                content_type=content_type,
            )
        )

    failed = application.save_multiple(iter(messages))
    return SaveMultipleResponse(failed=rejected + [save_failure(x) for x in failed])


@files.get(
    "/{file_id}",
    response_class=StreamingResponse,
    summary="Download a file",
    description="Streams file contents with its stored content type. "
    "`Content-Disposition` carries the original file name.",
    responses={
        status.HTTP_200_OK: {
            "description": "File contents",
            "content": {"image/*": {}, "video/*": {}, "audio/*": {}},
            "headers": {
                "Content-Disposition": {
                    "description": "`inline` or `attachment` with the original file name",
                    "schema": {"type": "string"},
                },
                "Content-Length": {
                    "description": "File size in bytes",
                    "schema": {"type": "integer"},
                },
            },
        },
        **NOT_FOUND,
    },
)
def get_file(
    file_id: uuid.UUID,
    application: ApplicationDep,
    download: Annotated[
        bool,
        Query(
            description="Send as attachment so browsers save the file instead of showing it"
        ),
    ] = False,
) -> StreamingResponse:
    try:
        stored = application.get_file(file_id)
    except FileNotFoundError:
        raise file_not_found(file_id) from None

    headers = {
        "Content-Disposition": content_disposition(stored.original_name, download)
    }
    if stored.size is not None:
        headers["Content-Length"] = str(stored.size)

    return StreamingResponse(
        stored.data, media_type=stored.content_type, headers=headers
    )


@files.get(
    "/{file_id}/info",
    summary="Get file info",
    responses=NOT_FOUND,
)
def get_file_info(file_id: uuid.UUID, application: ApplicationDep) -> FileInfo:
    try:
        return FileInfo.model_validate(application.file_storage.descriptor(file_id))
    except FileNotFoundError:
        raise file_not_found(file_id) from None


# --- search

search = APIRouter(prefix="/search", tags=["search"])


@search.get(
    "",
    summary="Search by a text query",
    description="Shorthand for `POST /search/query` with a single query.",
)
def search_files(
    application: ApplicationDep,
    q: Annotated[str, Query(min_length=1, description="Text query")],
    count: Annotated[
        int, Query(ge=1, le=MAX_COUNT, description="Number of files to return")
    ] = DEFAULT_COUNT,
) -> FileList:
    return file_list(application.get_by_queries([q], count)[0])


@search.post(
    "/query",
    summary="Search by text queries",
    description="Finds the closest files for each query, matching both file contents and text descriptions.",
)
def search_by_query(
    body: QuerySearchRequest, application: ApplicationDep
) -> SearchResponse:
    return search_response(application.get_by_queries(body.queries, body.count))


@search.post(
    "/embeddings",
    summary="Search by embeddings",
    description="Finds the closest files for each precomputed query embedding. "
    "Embeddings must have the dimension of the server model, otherwise the response is `422`.",
)
def search_by_embeddings(
    body: EmbeddingSearchRequest, application: ApplicationDep
) -> SearchResponse:
    vectors = [np.asarray(x, dtype=np.float32) for x in body.embeddings]
    try:
        return search_response(application.get_by_embeddings(vectors, body.count))
    except WrongEmbeddingsError:
        raise wrong_embeddings("embeddings") from None


# --- service

service = APIRouter(tags=["service"])


# Async so it answers even when every worker thread is busy with the model
@service.get("/health", summary="Health check")
async def health() -> Health:
    return Health()
