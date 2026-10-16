from .proto import service_pb2_grpc, service_pb2 as pb2
import grpc
from app import Application, FileDescriptor, NegativeRecordCount, UnsupportedTypeError, WrongEmbeddingsError
from typing import Callable, Iterator
import numpy as np
import uuid
from .config import Configuration

WRONG_EMBEDDINGS = "embedding dimension does not match the model"


class ProtocolError(Exception):
    pass


def unsupported_type_message(e: UnsupportedTypeError) -> str:
    content_type = e.args[0] if e.args else None
    if content_type is None: return "can not detect content type by file name, pass content_type"
    return f"unsupported content type {content_type}, expected image/*, video/* or audio/*"


class MemeStorageService(service_pb2_grpc.MemeStorageServicer):
    def __init__(self, app: Application, config: Configuration) -> None:
        super().__init__()
        self.app = app
        self.maximum_records: int = config.max_records

    @staticmethod
    def _chunk_uploaded_file(request: Iterator[pb2.SaveRequest]) -> Iterator[bytes]:
        for r in request:
            if r.WhichOneof("payload") != "data":
                raise ProtocolError("expected data chunk after info")
            yield r.data

    def Save(self, request_iterator: Iterator[pb2.SaveRequest], context: grpc.ServicerContext) -> pb2.SaveResponse:
        first = next(request_iterator, None)
        if first is None or first.WhichOneof("payload") != "info":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first message must be info")

        info = first.info
        data = MemeStorageService._chunk_uploaded_file(request_iterator)

        try:
            id = self.app.save(
                original_name=info.filename,
                data=data,
                content_type=info.content_type if info.content_type != "" else None,
                text_description=info.text_description if info.text_description != "" else None,

                embeddings=np.array(info.embeddings.value) if len(info.embeddings.value) != 0 else None,
                text_embeddings=np.array(info.text_embeddings.value) if len(info.text_embeddings.value) != 0 else None
            )
        except ProtocolError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except UnsupportedTypeError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, unsupported_type_message(e))
        except WrongEmbeddingsError:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, WRONG_EMBEDDINGS)

        return pb2.SaveResponse(uuid=str(id))

    def GetFile(self, request: pb2.UUIDRequest, context: grpc.ServicerContext) -> Iterator[pb2.DownloadResponse]:
        try:
            file_id = uuid.UUID(request.uuid)
        except ValueError:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid uuid: {request.uuid!r}")

        try:
            res = self.app.get_file(file_id)
        except FileNotFoundError:
            # Message may hold a server path, so the client gets only the id
            context.abort(grpc.StatusCode.NOT_FOUND, f"file not found: {file_id}")

        yield pb2.DownloadResponse(
            info=pb2.DownloadFileInfo(
                content_type=res.content_type,
                original_name=res.original_name,
                size=res.size
            )
        )

        for b in res.data:
            yield pb2.DownloadResponse(chunk=b)

    @staticmethod
    def _multiple_to_proto(items: list[list[FileDescriptor]]) -> pb2.MultipleIdResponse:
        res = [
            [str(x.id) for x in row]
            for row in items
        ]

        return pb2.MultipleIdResponse(result=[
            pb2.MultipleIds(ids=x) for x in res
        ])

    # Upper bound is an API limit from the config, lower bound is checked by the application
    def _search(self, record_count: int, search: Callable[[], list[list[FileDescriptor]]], context: grpc.ServicerContext) -> pb2.MultipleIdResponse:
        if record_count > self.maximum_records:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"record_count must be at most {self.maximum_records}, got {record_count}")

        try:
            found = search()
        except NegativeRecordCount:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"record_count must be at least 1, got {record_count}")
        except WrongEmbeddingsError:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, WRONG_EMBEDDINGS)
        except FileNotFoundError as e:
            # Vector storage returned ids that file storage does not know, storages are out of sync
            context.abort(grpc.StatusCode.INTERNAL, f"found files are missing from file storage: {e}")

        return MemeStorageService._multiple_to_proto(found)

    def GetByEmbeddings(self, request: pb2.EmbeddingRequest, context: grpc.ServicerContext) -> pb2.MultipleIdResponse:
        if not request.embeddings:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "at least one embedding is required")

        return self._search(request.record_count, lambda: self.app.get_by_embeddings(
            [np.array(x.value) for x in request.embeddings],
            request.record_count
        ), context)

    def GetByQuery(self, request: pb2.QueryRequest, context: grpc.ServicerContext) -> pb2.MultipleIdResponse:
        if not request.queries:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "at least one query is required")

        return self._search(request.record_count, lambda: self.app.get_by_queries(
            [x for x in request.queries],
            request.record_count
        ), context)
