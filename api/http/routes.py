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
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import (
    THUMBNAIL_CONTENT_TYPE,
    Application,
    EmbeddingsForbiddenError,
    FailureKind,
    FailureReport,
    FileDescriptor,
    Principal,
    QueueFullError,
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
    JobAccepted,
    JobInfo,
    JsonVector,
    QuerySearchRequest,
    SaveFailure,
    SaveFailureKind,
    SaveMultipleResponse,
    SearchResponse,
    SubmitMultipleResponse,
    SubmittedFile,
)

CHUNK_SIZE = 64 * 1024
# Stored files never change under their id, so clients and proxies may keep them forever
IMMUTABLE = "public, max-age=31536000, immutable"


class MediaResponse(FileResponse):
    # Every chunk is a read in a worker thread, bigger chunks mean fewer of them
    chunk_size = 1024 * 1024


MEDIA_PREFIXES = ("image/", "video/", "audio/")
# Sent by browsers and curl when they do not know the type, server guesses it by file name then
GENERIC_CONTENT_TYPES = ("", "application/octet-stream")

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": Error, "description": "File not found"}
}

FAILURE_KINDS: dict[FailureKind, SaveFailureKind] = {
    FailureKind.FILE_STORAGE: "file_storage",
    FailureKind.THUMBNAIL: "thumbnail",
    FailureKind.EMBEDDING_MODEL: "embedding_model",
    FailureKind.VECTOR_STORAGE: "vector_storage",
    FailureKind.LIMIT_EXCEEDED: "limit_exceeded",
}

TOO_LARGE: dict[int | str, dict[str, Any]] = {
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": Error,
        "description": "The file is larger or longer than the server limit, nothing is stored",
    }
}


async def get_application(request: Request) -> Application:
    return request.app.state.application


ApplicationDep = Annotated[Application, Depends(get_application)]

bearer = HTTPBearer(auto_error=False, description="Token of a user or the admin token")


# Sync on purpose: finding a user is a database query, it runs in a worker thread
def authenticate(
    request: Request,
    application: ApplicationDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    principal = application.authenticate(
        credentials.credentials if credentials else None
    )
    if principal is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "missing or invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # The activity log reads it after the response is sent
    request.state.principal = principal
    return principal


PrincipalDep = Annotated[Principal, Depends(authenticate)]


def require_admin(principal: PrincipalDep) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin token required")
    return principal


UNAUTHORIZED: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {
        "model": Error,
        "description": "Missing, wrong or revoked token",
    }
}


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


# What only the application can find out and the client can fix: a file that is too big or too long,
# or one that can not be decoded. Wrong requests are checked before saving, so the rest is a server failure.
# 415 and not 422 for a damaged file: the detail of 422 is a list of field errors, clients parse it so
CLIENT_FAILURES = {
    FailureKind.LIMIT_EXCEEDED: status.HTTP_413_CONTENT_TOO_LARGE,
    FailureKind.THUMBNAIL: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}


def save_failed(report: FailureReport) -> HTTPException:
    code = CLIENT_FAILURES.get(report.kind, status.HTTP_500_INTERNAL_SERVER_ERROR)
    return HTTPException(code, f"{report.kind.value}: {report.value}")


EMBEDDINGS_FORBIDDEN: dict[int | str, dict[str, Any]] = {
    status.HTTP_403_FORBIDDEN: {
        "model": Error,
        "description": "Ready embeddings were sent, but the user may not upload them",
    }
}


# Ready embeddings go into the index as they are, so only users the admin trusted may send them.
# Application reports a wrong dimension as a model failure, here it is a client error
def ready_embeddings(
    application: Application,
    principal: Principal,
    embeddings: list[float] | None,
    text_embeddings: list[float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if embeddings is None and text_embeddings is None:
        return None, None

    try:
        application.check_embeddings(principal)
    except EmbeddingsForbiddenError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e)) from None

    checked = []
    for field, value in (
        ("embeddings", embeddings),
        ("text_embeddings", text_embeddings),
    ):
        vector = None if value is None else np.asarray(value, dtype=np.float32)
        if (
            vector is not None
            and not application.embedding_provider.validate_embedding_dimension(vector)
        ):
            raise wrong_embeddings(field)
        checked.append(vector)
    return checked[0], checked[1]


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


# With save strategy "fast" the model refuses a non-media file for the whole batch,
# so such files are not passed to the application
def batch_messages(
    uploads: list[UploadFile], principal: Principal
) -> tuple[list[SaveMessage], list[SaveFailure]]:
    if not all(x.filename for x in uploads):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "every file must have a name")

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
                owner_id=principal.user_id,
            )
        )
    return messages, rejected


def file_list(descriptors: list[FileDescriptor]) -> FileList:
    return FileList(items=[FileInfo.model_validate(x) for x in descriptors])


def search_response(found: list[list[FileDescriptor]]) -> SearchResponse:
    return SearchResponse(results=[file_list(row) for row in found])


# --- files

files = APIRouter(
    prefix="/files",
    tags=["files"],
    dependencies=[Depends(authenticate)],
    responses=UNAUTHORIZED,
)


@files.post(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Upload a file",
    description="Stores an image, video or audio file and indexes it for search. "
    "Content type is taken from the file part, or guessed by file name when it is missing or `application/octet-stream`. "
    "A thumbnail is built right after the file is stored, a file it can not be built for is not stored. "
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
            "description": "File is not an image, video or audio, or can not be decoded: it is damaged or not what its type says",
        },
        **EMBEDDINGS_FORBIDDEN,
        **TOO_LARGE,
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": Error,
            "description": "File storage, embedding model or vector storage failed, nothing is stored",
        },
    },
)
def save_file(
    application: ApplicationDep,
    principal: PrincipalDep,
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

    vector, text_vector = ready_embeddings(
        application, principal, embeddings, text_embeddings
    )
    msg = SaveMessage(
        original_name=file.filename,
        data=file_data(file),
        content_type=media_type(file),
        text_description=description,
        embeddings=vector,
        text_embeddings=text_vector,
        owner_id=principal.user_id,
    )

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
    principal: PrincipalDep,
    # Alias keeps the form field name without shadowing the router
    uploads: Annotated[
        list[UploadFile],
        File(alias="files", description="Image, video or audio files"),
    ],
) -> SaveMultipleResponse:
    messages, rejected = batch_messages(uploads, principal)
    failed = application.save_multiple(iter(messages))
    return SaveMultipleResponse(failed=rejected + [save_failure(x) for x in failed])


@files.get(
    "/{file_id}",
    response_class=FileResponse,
    summary="Download a file",
    description="Sends file contents with its stored content type, byte ranges are supported. "
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
) -> FileResponse:
    try:
        descriptor = application.file_storage.descriptor(file_id)
    except FileNotFoundError:
        raise file_not_found(file_id) from None

    path = application.file_storage.get_full_path(file_id)
    if not path.is_file():
        raise file_not_found(file_id)

    # Content-Length, ETag and byte ranges are added by the response itself
    return MediaResponse(
        path,
        media_type=descriptor.content_type,
        headers={
            "Content-Disposition": content_disposition(
                descriptor.original_name, download
            ),
            "Cache-Control": IMMUTABLE,
            # Content type is chosen by the uploader, an SVG must not run scripts on this origin
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
        },
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


@files.get(
    "/{file_id}/thumbnail",
    response_class=Response,
    summary="Get file thumbnail",
    description="JPEG thumbnail of the file, the longest side is `thumbnails.size` from the server config. "
    "Audio and files stored before thumbnails appeared get a placeholder of their type.",
    responses={
        status.HTTP_200_OK: {
            "description": "Thumbnail",
            "content": {THUMBNAIL_CONTENT_TYPE: {}},
        },
        **NOT_FOUND,
    },
)
def get_thumbnail(file_id: uuid.UUID, application: ApplicationDep) -> Response:
    try:
        thumbnail = application.get_thumbnail(file_id)
    except FileNotFoundError:
        raise file_not_found(file_id) from None

    return Response(thumbnail, media_type=THUMBNAIL_CONTENT_TYPE)


# --- jobs

jobs = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(authenticate)],
    responses=UNAUTHORIZED,
)

QUEUE_FULL: dict[int | str, dict[str, Any]] = {
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "model": Error,
        "description": "The queue is full, try again later",
    }
}


@jobs.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a file and index it later",
    description="Stores the file and answers at once, embedding runs in the background. "
    "Poll `GET /jobs/{id}`: the file is searchable when the status is `done`. "
    "Precomputed embeddings need the right to send them, as in `POST /files`. "
    "Jobs that were queued survive a server restart.",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": Error,
            "description": "File part has no file name",
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "model": Error,
            "description": "File is not an image, video or audio, or can not be decoded: it is damaged or not what its type says",
        },
        **EMBEDDINGS_FORBIDDEN,
        **TOO_LARGE,
        **QUEUE_FULL,
    },
)
def submit_file(
    application: ApplicationDep,
    principal: PrincipalDep,
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
) -> JobAccepted:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file must have a name")

    vector, text_vector = ready_embeddings(
        application, principal, embeddings, text_embeddings
    )
    msg = SaveMessage(
        original_name=file.filename,
        data=file_data(file),
        content_type=media_type(file),
        text_description=description,
        embeddings=vector,
        text_embeddings=text_vector,
        owner_id=principal.user_id,
    )

    try:
        result = application.submit(msg)
    except QueueFullError:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "queue is full"
        ) from None

    if isinstance(result, FailureReport):
        raise save_failed(result)
    return JobAccepted(id=result.id)


@jobs.post(
    "/batch",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload several files and index them later",
    description="Like `POST /jobs` for many files. A file that is refused, because of its type, size, "
    "length or a full queue, goes to `failed` and does not fail the request.",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": Error,
            "description": "A file part has no file name, nothing is stored",
        },
    },
)
def submit_files(
    application: ApplicationDep,
    principal: PrincipalDep,
    uploads: Annotated[
        list[UploadFile],
        File(alias="files", description="Image, video or audio files"),
    ],
) -> SubmitMultipleResponse:
    messages, rejected = batch_messages(uploads, principal)
    queued, failed = application.submit_multiple(iter(messages))
    return SubmitMultipleResponse(
        jobs=[SubmittedFile(id=j.id, filename=j.original_name) for j in queued],
        failed=rejected + [save_failure(x) for x in failed],
    )


@jobs.get(
    "/{job_id}",
    summary="Get job status",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": Error, "description": "Job not found"}
    },
)
def get_job(job_id: uuid.UUID, application: ApplicationDep) -> JobInfo:
    job = application.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job not found: {job_id}")

    error = None
    if job.error_kind is not None:
        error = SaveFailure(
            kind=FAILURE_KINDS[job.error_kind], message=job.error_message or ""
        )
    return JobInfo(
        id=job.id,
        original_name=job.original_name,
        status=job.status.value,
        error=error,
    )


# --- search

search = APIRouter(
    prefix="/search",
    tags=["search"],
    dependencies=[Depends(authenticate)],
    responses=UNAUTHORIZED,
)


@search.get(
    "",
    summary="Search by a text query",
    description="Shorthand for `POST /search/query` with a single query.",
)
def search_files(
    request: Request,
    application: ApplicationDep,
    q: Annotated[str, Query(min_length=1, description="Text query")],
    count: Annotated[
        int, Query(ge=1, le=MAX_COUNT, description="Number of files to return")
    ] = DEFAULT_COUNT,
) -> FileList:
    request.state.detail = q
    return file_list(application.get_by_queries([q], count)[0])


@search.post(
    "/query",
    summary="Search by text queries",
    description="Finds the closest files for each query, matching both file contents and text descriptions.",
)
def search_by_query(
    request: Request, body: QuerySearchRequest, application: ApplicationDep
) -> SearchResponse:
    request.state.detail = " | ".join(body.queries)
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
