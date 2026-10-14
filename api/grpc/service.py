from .proto import service_pb2_grpc, service_pb2 as pb2
import grpc
from app import Application, FileDescriptor
from typing import Iterator, cast
import numpy as np
import uuid
from typing import Iterator

class MemeStorageService(service_pb2_grpc.MemeStorageServicer):
    def __init__(self, app: Application) -> None:
        super().__init__()
        self.app = app

    @staticmethod
    def _chunk_uploaded_file(request: Iterator[pb2.SaveRequest]) -> Iterator[bytes]:
        for r in request: 
            if r.WhichOneof("payload") != "data":
                raise ValueError("expected data chunk")
            yield r.data

    def Save(self, request_iterator: Iterator[pb2.SaveRequest], context: grpc.ServicerContext) -> pb2.SaveResponse:
        first = next(request_iterator, None)
        if first is None or first.WhichOneof("payload") != "info":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first message must be info")

        info = first.info
        data = MemeStorageService._chunk_uploaded_file(request_iterator)

        id = self.app.save(
            original_name=info.filename,
            data=data,
            content_type=info.content_type if info.content_type != "" else None,
            text_description=info.text_description if info.text_description != "" else None,

            embeddings=np.array(info.embeddings.value) if len(info.embeddings.value) != 0 else None,
            text_embeddings=np.array(info.text_embeddings.value) if len(info.text_embeddings.value) != 0 else None
        )

        return pb2.SaveResponse(uuid=str(id))

    def GetFile(self, request: pb2.UUIDRequest, context: grpc.ServicerContext) -> Iterator[pb2.DownloadResponse]:
        res = self.app.get_file(uuid.UUID(request.uuid))

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

    def GetByEmbeddings(self, request: pb2.EmbeddingRequest, context: grpc.ServicerContext) -> pb2.MultipleIdResponse:
        res = self.app.get_by_embeddings(
            [np.array(x.value) for x in request.embeddings], 
            request.record_count
        )

        return MemeStorageService._multiple_to_proto(res)


    def GetByQuery(self, request: pb2.QueryRequest, context: grpc.ServicerContext) -> pb2.MultipleIdResponse:
        res = self.app.get_by_queries(
            [x for x in request.queries],
            request.record_count
        )
        return MemeStorageService._multiple_to_proto(res)