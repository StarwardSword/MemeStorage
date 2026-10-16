import uuid
from functools import partial
from mimetypes import guess_type
from typing import Annotated, Any
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

from app import Application, FileDescriptor, WrongEmbeddingsError
from .schemas import (
    DEFAULT_COUNT,
    MAX_COUNT,
    JsonVector,
    EmbeddingSearchRequest,
    Error,
    FileInfo,
    FileList,
    Health,
    QuerySearchRequest,
    SaveResponse,
    SearchResponse,
)

CHUNK_SIZE = 64 * 1024
MEDIA_PREFIXES = ("image/", "video/", "audio/")
# Sent by browsers and curl when they do not know the type, server guesses it by file name then
GENERIC_CONTENT_TYPES = ("", "application/octet-stream")

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": Error, "description": "File not found"}
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


def media_type(file: UploadFile) -> str:
    content_type = file.content_type
    if content_type is None or content_type in GENERIC_CONTENT_TYPES:
        content_type = guess_type(file.filename or "")[0]

    if content_type is None or not content_type.startswith(MEDIA_PREFIXES):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported content type {content_type}, expected image/*, video/* or audio/*",
        )
    return content_type


# filename* carries the real name (RFC 6266), filename is an ASCII fallback for old clients
def content_disposition(filename: str, attachment: bool) -> str:
    fallback = "".join(
        c if c.isascii() and c.isprintable() and c not in '"\\' else "_"
        for c in filename
    )
    return f"{'attachment' if attachment else 'inline'}; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"


def file_list(descriptors: list[FileDescriptor]) -> FileList:
    return FileList(items=[FileInfo.model_validate(x) for x in descriptors])


def search_response(found: list[list[FileDescriptor]]) -> SearchResponse:
    return SearchResponse(results=[file_list(row) for row in found])


# --- files

files = APIRouter(prefix="/files", tags=["files"])


@files.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file",
    description="Stores an image, video or audio file and indexes it for search. "
    "Content type is taken from the file part, or guessed by file name when it is missing or `application/octet-stream`. "
    "`embeddings` must have the dimension of the server model, otherwise the response is `422`.",
    response_description="File stored",
    responses={
        status.HTTP_201_CREATED: {
            "headers": {
                "Location": {
                    "description": "URL of the stored file",
                    "schema": {"type": "string", "format": "uri"},
                },
            }
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": Error,
            "description": "File part has no file name",
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "model": Error,
            "description": "File is not an image, video or audio",
        },
    },
)
def save_file(
    request: Request,
    response: Response,
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
) -> SaveResponse:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file must have a name")

    try:
        file_id = application.save(
            original_name=file.filename,
            data=iter(partial(file.file.read, CHUNK_SIZE), b""),
            content_type=media_type(file),
            text_description=description,
            embeddings=np.asarray(embeddings, dtype=np.float32)
            if embeddings is not None
            else None,
            text_embeddings=np.asarray(text_embeddings, dtype=np.float32)
            if text_embeddings is not None
            else None,
        )
    except WrongEmbeddingsError:
        raise wrong_embeddings("embeddings") from None

    response.headers["Location"] = str(request.url_for("get_file", file_id=file_id))
    return SaveResponse(id=file_id)


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
