import time
import uuid
from mimetypes import guess_type
from pathlib import Path
from typing import Callable, Iterator, TypeIs

import grpc
import numpy as np

from app import (
    THUMBNAIL_CONTENT_TYPE,
    ActivityDisabledError,
    Application,
    EmbeddingsForbiddenError,
    FailureKind,
    FailureReport,
    FileDescriptor,
    JobStatus,
    NegativeRecordCount,
    Principal,
    QueueFullError,
    SaveMessage,
    User,
    UserExistsError,
    UserNotFoundError,
    WrongEmbeddingsError,
)

from .activity import remember
from .config import Configuration
from .proto import service_pb2 as pb2
from .proto import service_pb2_grpc

WRONG_EMBEDDINGS = "embedding dimension does not match the model"
BEARER = "bearer "
MEDIA_PREFIXES = ("image/", "video/", "audio/")

FAILURE_KINDS = {
    FailureKind.FILE_STORAGE: pb2.SaveFailure.FILE_STORAGE,
    FailureKind.THUMBNAIL: pb2.SaveFailure.THUMBNAIL,
    FailureKind.EMBEDDING_MODEL: pb2.SaveFailure.EMBEDDING_MODEL,
    FailureKind.VECTOR_STORAGE: pb2.SaveFailure.VECTOR_STORAGE,
    FailureKind.LIMIT_EXCEEDED: pb2.SaveFailure.LIMIT_EXCEEDED,
}

JOB_STATUSES = {
    JobStatus.PENDING: pb2.JobResponse.PENDING,
    JobStatus.RUNNING: pb2.JobResponse.RUNNING,
    JobStatus.DONE: pb2.JobResponse.DONE,
    JobStatus.FAILED: pb2.JobResponse.FAILED,
}


class ProtocolError(Exception):
    pass


def unsupported_type_message(content_type: str | None) -> str:
    if content_type is None:
        return "can not detect content type by file name, pass content_type"
    return (
        f"unsupported content type {content_type}, expected image/*, video/* or audio/*"
    )


def detect_content_type(info: pb2.SaveFileInfo) -> str | None:
    return info.content_type or guess_type(Path(info.filename).name)[0]


def is_media(content_type: str | None) -> TypeIs[str]:
    return content_type is not None and content_type.startswith(MEDIA_PREFIXES)


def failure_message(report: FailureReport) -> str:
    return f"{report.kind.value}: {report.value}"


def failure_code(report: FailureReport) -> grpc.StatusCode:
    if report.kind is FailureKind.LIMIT_EXCEEDED:
        return grpc.StatusCode.RESOURCE_EXHAUSTED
    return grpc.StatusCode.INTERNAL


def failure_to_proto(report: FailureReport) -> pb2.SaveFailure:
    return pb2.SaveFailure(kind=FAILURE_KINDS[report.kind], message=report.value)


def user_to_proto(user: User) -> pb2.UserInfo:
    return pb2.UserInfo(
        uuid=str(user.id),
        name=user.name,
        allow_embeddings=user.allow_embeddings,
        created_at=user.created_at,
        deleted_at=user.deleted_at,
    )


def has_embeddings(info: pb2.SaveFileInfo) -> bool:
    return len(info.embeddings.value) != 0 or len(info.text_embeddings.value) != 0


def save_message(
    info: pb2.SaveFileInfo,
    data: Iterator[bytes],
    content_type: str,
    owner_id: uuid.UUID | None,
) -> SaveMessage:
    return SaveMessage(
        original_name=info.filename,
        data=data,
        content_type=content_type,
        owner_id=owner_id,
        text_description=info.text_description if info.text_description != "" else None,
        embeddings=np.array(info.embeddings.value, dtype=np.float32)
        if len(info.embeddings.value) != 0
        else None,
        text_embeddings=np.array(info.text_embeddings.value, dtype=np.float32)
        if len(info.text_embeddings.value) != 0
        else None,
    )


class MemeStorageService(service_pb2_grpc.MemeStorageServicer):
    def __init__(self, app: Application, config: Configuration) -> None:
        super().__init__()
        self.app = app
        self.maximum_records: int = config.max_records

    # Who calls, from the authorization metadata. Aborts the call when there is nobody
    def _auth(self, context: grpc.ServicerContext, admin: bool = False) -> Principal:
        header = dict(context.invocation_metadata() or ()).get("authorization")
        token = (
            header[len(BEARER) :]
            if isinstance(header, str) and header.lower().startswith(BEARER)
            else None
        )
        principal = self.app.authenticate(token)
        if principal is None:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing or invalid token")
        remember(principal=principal)
        if admin and not principal.is_admin:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "admin token required")
        return principal

    # Why this caller may not send ready embeddings, None when it may
    def _embeddings_refusal(self, principal: Principal) -> str | None:
        try:
            self.app.check_embeddings(principal)
        except EmbeddingsForbiddenError as e:
            return str(e)
        return None

    # Application reports a failed read as a file storage failure, so the protocol error is kept aside
    @staticmethod
    def _chunk_uploaded_file(
        request: Iterator[pb2.SaveRequest], errors: list[ProtocolError]
    ) -> Iterator[bytes]:
        for r in request:
            if r.WhichOneof("payload") != "data":
                errors.append(ProtocolError("expected data chunk after info"))
                raise errors[-1]
            yield r.data

    # First message of a single file upload is its info, the rest is read lazily while it is stored
    def _read_upload(
        self,
        principal: Principal,
        request_iterator: Iterator[pb2.SaveRequest],
        context: grpc.ServicerContext,
    ) -> tuple[SaveMessage, list[ProtocolError]]:
        first = next(request_iterator, None)
        if first is None or first.WhichOneof("payload") != "info":
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "first message must be info"
            )

        info = first.info
        content_type = detect_content_type(info)
        if not is_media(content_type):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, unsupported_type_message(content_type)
            )

        if has_embeddings(info):
            refusal = self._embeddings_refusal(principal)
            if refusal is not None:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, refusal)

        errors: list[ProtocolError] = []
        msg = save_message(
            info,
            MemeStorageService._chunk_uploaded_file(request_iterator, errors),
            content_type,
            principal.user_id,
        )

        # Application reports a wrong dimension as a model failure, here it is a client error
        if not all(
            self.app.embedding_provider.validate_embedding_dimension(x)
            for x in (msg.embeddings, msg.text_embeddings)
            if x is not None
        ):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, WRONG_EMBEDDINGS)
        return msg, errors

    def Save(
        self, request_iterator: Iterator[pb2.SaveRequest], context: grpc.ServicerContext
    ) -> pb2.SaveResponse:
        principal = self._auth(context)
        msg, errors = self._read_upload(principal, request_iterator, context)

        failure = self.app.save(msg)

        if errors:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(errors[0]))
        if failure is not None:
            context.abort(failure_code(failure), failure_message(failure))

        return pb2.SaveResponse()

    def Submit(
        self, request_iterator: Iterator[pb2.SaveRequest], context: grpc.ServicerContext
    ) -> pb2.SubmitResponse:
        principal = self._auth(context)
        msg, errors = self._read_upload(principal, request_iterator, context)

        try:
            result = self.app.submit(msg)
        except QueueFullError:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "queue is full")

        if errors:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(errors[0]))
        if isinstance(result, FailureReport):
            context.abort(failure_code(result), failure_message(result))

        return pb2.SubmitResponse(uuid=str(result.id))

    # Every info message starts the next file. With save_strategy "fast" the model refuses
    # a non-media file for the whole batch, so such files are not passed to the application
    @staticmethod
    def _split_uploaded_files(
        first: pb2.SaveFileInfo,
        request: Iterator[pb2.SaveMultipleRequest],
        rejected: list[pb2.SaveFailure],
        embeddings_refusal: str | None,
        owner_id: uuid.UUID | None,
    ) -> Iterator[SaveMessage]:
        info: pb2.SaveFileInfo | None = first
        while info is not None:
            current, info = info, None

            def chunks() -> Iterator[bytes]:
                nonlocal info
                for r in request:
                    if r.WhichOneof("payload") == "info":
                        info = r.info
                        return
                    yield r.data

            data = chunks()
            content_type = detect_content_type(current)
            if has_embeddings(current) and embeddings_refusal is not None:
                rejected.append(
                    pb2.SaveFailure(
                        kind=pb2.SaveFailure.FORBIDDEN,
                        message=f"{current.filename}: {embeddings_refusal}",
                    )
                )
            elif is_media(content_type):
                yield save_message(current, data, content_type, owner_id)
            else:
                rejected.append(
                    pb2.SaveFailure(
                        kind=pb2.SaveFailure.UNSUPPORTED_TYPE,
                        message=f"{current.filename}: {unsupported_type_message(content_type)}",
                    )
                )

            # File storage may stop reading a file halfway on error, the rest of it is skipped
            for _ in data:
                pass

    def SaveMultiple(
        self,
        request_iterator: Iterator[pb2.SaveMultipleRequest],
        context: grpc.ServicerContext,
    ) -> pb2.SaveMultipleResponse:
        principal = self._auth(context)
        first = next(request_iterator, None)
        if first is None or first.WhichOneof("payload") != "info":
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "first message must be info"
            )

        rejected: list[pb2.SaveFailure] = []
        failed = self.app.save_multiple(
            MemeStorageService._split_uploaded_files(
                first.info,
                request_iterator,
                rejected,
                self._embeddings_refusal(principal),
                principal.user_id,
            )
        )

        return pb2.SaveMultipleResponse(
            failed=rejected + [failure_to_proto(x) for x in failed]
        )

    def SubmitMultiple(
        self,
        request_iterator: Iterator[pb2.SaveMultipleRequest],
        context: grpc.ServicerContext,
    ) -> pb2.SubmitMultipleResponse:
        principal = self._auth(context)
        first = next(request_iterator, None)
        if first is None or first.WhichOneof("payload") != "info":
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "first message must be info"
            )

        rejected: list[pb2.SaveFailure] = []
        queued, failed = self.app.submit_multiple(
            MemeStorageService._split_uploaded_files(
                first.info,
                request_iterator,
                rejected,
                self._embeddings_refusal(principal),
                principal.user_id,
            )
        )

        return pb2.SubmitMultipleResponse(
            jobs=[
                pb2.SubmittedFile(uuid=str(j.id), filename=j.original_name)
                for j in queued
            ],
            failed=rejected + [failure_to_proto(x) for x in failed],
        )

    def GetJob(
        self, request: pb2.UUIDRequest, context: grpc.ServicerContext
    ) -> pb2.JobResponse:
        self._auth(context)
        try:
            job_id = uuid.UUID(request.uuid)
        except ValueError:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"invalid uuid: {request.uuid!r}"
            )

        job = self.app.get_job(job_id)
        if job is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"job not found: {job_id}")

        error = None
        if job.error_kind is not None:
            error = pb2.SaveFailure(
                kind=FAILURE_KINDS[job.error_kind], message=job.error_message or ""
            )
        return pb2.JobResponse(
            status=JOB_STATUSES[job.status],
            original_name=job.original_name,
            error=error,
        )

    @staticmethod
    def _file_id(request: pb2.UUIDRequest, context: grpc.ServicerContext) -> uuid.UUID:
        try:
            return uuid.UUID(request.uuid)
        except ValueError:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"invalid uuid: {request.uuid!r}"
            )

    def GetFile(
        self, request: pb2.UUIDRequest, context: grpc.ServicerContext
    ) -> Iterator[pb2.DownloadResponse]:
        self._auth(context)
        file_id = MemeStorageService._file_id(request, context)

        try:
            res = self.app.get_file(file_id)
        except FileNotFoundError:
            # Message may hold a server path, so the client gets only the id
            context.abort(grpc.StatusCode.NOT_FOUND, f"file not found: {file_id}")

        yield pb2.DownloadResponse(
            info=pb2.DownloadFileInfo(
                content_type=res.content_type,
                original_name=res.original_name,
                size=res.size,
            )
        )

        for b in res.data:
            yield pb2.DownloadResponse(chunk=b)

    def GetThumbnail(
        self, request: pb2.UUIDRequest, context: grpc.ServicerContext
    ) -> pb2.ThumbnailResponse:
        self._auth(context)
        file_id = MemeStorageService._file_id(request, context)

        try:
            thumbnail = self.app.get_thumbnail(file_id)
        except FileNotFoundError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"file not found: {file_id}")

        return pb2.ThumbnailResponse(
            content_type=THUMBNAIL_CONTENT_TYPE, data=thumbnail
        )

    @staticmethod
    def _multiple_to_proto(items: list[list[FileDescriptor]]) -> pb2.MultipleIdResponse:
        res = [[str(x.id) for x in row] for row in items]

        return pb2.MultipleIdResponse(result=[pb2.MultipleIds(ids=x) for x in res])

    # Upper bound is an API limit from the config, lower bound is checked by the application
    def _search(
        self,
        record_count: int,
        search: Callable[[], list[list[FileDescriptor]]],
        context: grpc.ServicerContext,
    ) -> pb2.MultipleIdResponse:
        if record_count > self.maximum_records:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"record_count must be at most {self.maximum_records}, got {record_count}",
            )

        try:
            found = search()
        except NegativeRecordCount:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"record_count must be at least 1, got {record_count}",
            )
        except WrongEmbeddingsError:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, WRONG_EMBEDDINGS)
        except FileNotFoundError as e:
            # Vector storage returned ids that file storage does not know, storages are out of sync
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"found files are missing from file storage: {e}",
            )

        return MemeStorageService._multiple_to_proto(found)

    def GetByEmbeddings(
        self, request: pb2.EmbeddingRequest, context: grpc.ServicerContext
    ) -> pb2.MultipleIdResponse:
        self._auth(context)
        if not request.embeddings:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "at least one embedding is required"
            )

        return self._search(
            request.record_count,
            lambda: self.app.get_by_embeddings(
                [np.array(x.value, dtype=np.float32) for x in request.embeddings],
                request.record_count,
            ),
            context,
        )

    def GetByQuery(
        self, request: pb2.QueryRequest, context: grpc.ServicerContext
    ) -> pb2.MultipleIdResponse:
        self._auth(context)
        remember(detail=" | ".join(request.queries)[:500])
        if not request.queries:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "at least one query is required"
            )

        return self._search(
            request.record_count,
            lambda: self.app.get_by_queries(
                [x for x in request.queries], request.record_count
            ),
            context,
        )

    # --- admin

    def _user_id(self, text: str, context: grpc.ServicerContext) -> uuid.UUID:
        try:
            return uuid.UUID(text)
        except ValueError:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid uuid: {text!r}")

    def CreateUser(
        self, request: pb2.CreateUserRequest, context: grpc.ServicerContext
    ) -> pb2.CreateUserResponse:
        self._auth(context, admin=True)
        try:
            user, token = self.app.create_user(request.name, request.allow_embeddings)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except UserExistsError:
            context.abort(
                grpc.StatusCode.ALREADY_EXISTS, f"user already exists: {request.name}"
            )
        return pb2.CreateUserResponse(user=user_to_proto(user), token=token)

    def ListUsers(
        self, request: pb2.Empty, context: grpc.ServicerContext
    ) -> pb2.ListUsersResponse:
        self._auth(context, admin=True)
        return pb2.ListUsersResponse(
            users=[user_to_proto(u) for u in self.app.list_users()]
        )

    def UpdateUser(
        self, request: pb2.UpdateUserRequest, context: grpc.ServicerContext
    ) -> pb2.UserInfo:
        self._auth(context, admin=True)
        user_id = self._user_id(request.uuid, context)
        try:
            return user_to_proto(
                self.app.update_user(user_id, request.allow_embeddings)
            )
        except UserNotFoundError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"user not found: {user_id}")

    def DeleteUser(
        self, request: pb2.UUIDRequest, context: grpc.ServicerContext
    ) -> pb2.Empty:
        self._auth(context, admin=True)
        user_id = self._user_id(request.uuid, context)
        try:
            self.app.delete_user(user_id)
        except UserNotFoundError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"user not found: {user_id}")
        return pb2.Empty()

    def RotateToken(
        self, request: pb2.UUIDRequest, context: grpc.ServicerContext
    ) -> pb2.TokenResponse:
        self._auth(context, admin=True)
        user_id = self._user_id(request.uuid, context)
        try:
            token, restored = self.app.rotate_token(user_id)
            return pb2.TokenResponse(token=token, restored=restored)
        except UserNotFoundError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"user not found: {user_id}")

    def GetStats(
        self, request: pb2.Empty, context: grpc.ServicerContext
    ) -> pb2.StatsResponse:
        self._auth(context, admin=True)
        stats = self.app.stats()
        return pb2.StatsResponse(
            users=[
                pb2.UserStats(
                    uuid=str(u.user_id) if u.user_id else "",
                    name=u.name or "",
                    deleted=u.deleted,
                    files=u.files,
                    bytes=u.bytes,
                    by_type={
                        k: pb2.TypeUsage(files=v.files, bytes=v.bytes)
                        for k, v in u.by_type.items()
                    },
                    last_upload=u.last_upload,
                )
                for u in stats.users
            ],
            files=stats.files,
            bytes=stats.bytes,
        )

    def GetActivity(
        self, request: pb2.ActivityRequest, context: grpc.ServicerContext
    ) -> pb2.ActivityResponse:
        self._auth(context, admin=True)
        until = request.until if request.HasField("until") else time.time()
        since = request.since if request.HasField("since") else until - 7 * 86400
        try:
            a = self.app.activity_summary(since, until)
        except ActivityDisabledError:
            context.abort(grpc.StatusCode.UNAVAILABLE, "activity log is turned off")

        return pb2.ActivityResponse(
            since=a.since,
            until=a.until,
            requests=a.requests,
            ok=a.ok,
            denied=a.denied,
            client_error=a.client_error,
            server_error=a.server_error,
            avg_ms=a.avg_ms,
            bytes_in=a.bytes_in,
            bytes_out=a.bytes_out,
            by_transport=a.by_transport,
            by_action=[pb2.ActionStats(**x.model_dump()) for x in a.by_action],
            by_actor=[
                pb2.ActorStats(
                    uuid=str(x.user_id) if x.user_id else "",
                    actor=x.actor or "",
                    requests=x.requests,
                    errors=x.errors,
                    bytes_in=x.bytes_in,
                    bytes_out=x.bytes_out,
                    first_seen=x.first_seen,
                    last_seen=x.last_seen,
                )
                for x in a.by_actor
            ],
            top_queries=[pb2.QueryCount(**x.model_dump()) for x in a.top_queries],
            jobs_by_status=a.jobs.by_status,
            job_failures=a.jobs.failures,
            dropped=a.dropped,
        )
