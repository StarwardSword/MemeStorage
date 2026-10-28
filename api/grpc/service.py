import uuid
from mimetypes import guess_type
from pathlib import Path
from typing import Callable, Iterator, TypeIs

import grpc
import numpy as np

from app import (
    THUMBNAIL_CONTENT_TYPE,
    Application,
    FailureKind,
    FailureReport,
    FileDescriptor,
    NegativeRecordCount,
    SaveMessage,
    WrongEmbeddingsError,
)

from .config import Configuration
from .proto import service_pb2 as pb2
from .proto import service_pb2_grpc

WRONG_EMBEDDINGS = "embedding dimension does not match the model"
MEDIA_PREFIXES = ("image/", "video/", "audio/")

FAILURE_KINDS = {
    FailureKind.FILE_STORAGE: pb2.SaveFailure.FILE_STORAGE,
    FailureKind.THUMBNAIL: pb2.SaveFailure.THUMBNAIL,
    FailureKind.EMBEDDING_MODEL: pb2.SaveFailure.EMBEDDING_MODEL,
    FailureKind.VECTOR_STORAGE: pb2.SaveFailure.VECTOR_STORAGE,
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


def failure_to_proto(report: FailureReport) -> pb2.SaveFailure:
    return pb2.SaveFailure(kind=FAILURE_KINDS[report.kind], message=report.value)


def save_message(
    info: pb2.SaveFileInfo, data: Iterator[bytes], content_type: str
) -> SaveMessage:
    return SaveMessage(
        original_name=info.filename,
        data=data,
        content_type=content_type,
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

    def Save(
        self, request_iterator: Iterator[pb2.SaveRequest], context: grpc.ServicerContext
    ) -> pb2.SaveResponse:
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

        errors: list[ProtocolError] = []
        msg = save_message(
            info,
            MemeStorageService._chunk_uploaded_file(request_iterator, errors),
            content_type,
        )

        # Application reports a wrong dimension as a model failure, here it is a client error
        if not all(
            self.app.embedding_provider.validate_embedding_dimension(x)
            for x in (msg.embeddings, msg.text_embeddings)
            if x is not None
        ):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, WRONG_EMBEDDINGS)

        failure = self.app.save(msg)

        if errors:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(errors[0]))
        if failure is not None:
            context.abort(grpc.StatusCode.INTERNAL, failure_message(failure))

        return pb2.SaveResponse()

    # Every info message starts the next file. With save_strategy "fast" the model refuses
    # a non-media file for the whole batch, so such files are not passed to the application
    @staticmethod
    def _split_uploaded_files(
        first: pb2.SaveFileInfo,
        request: Iterator[pb2.SaveMultipleRequest],
        rejected: list[pb2.SaveFailure],
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
            if is_media(content_type):
                yield save_message(current, data, content_type)
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
        first = next(request_iterator, None)
        if first is None or first.WhichOneof("payload") != "info":
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "first message must be info"
            )

        rejected: list[pb2.SaveFailure] = []
        failed = self.app.save_multiple(
            MemeStorageService._split_uploaded_files(
                first.info, request_iterator, rejected
            )
        )

        return pb2.SaveMultipleResponse(
            failed=rejected + [failure_to_proto(x) for x in failed]
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
