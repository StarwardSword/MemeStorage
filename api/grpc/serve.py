from concurrent import futures

import grpc

from app import Application

from .activity import ActivityInterceptor
from .config import Configuration
from .proto import service_pb2_grpc
from .service import MemeStorageService


# Returns started server without blocking, caller decides when to wait for or stop it
def serve(app: Application, config: Configuration) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        interceptors=[ActivityInterceptor(app.activity)],
    )
    service_pb2_grpc.add_MemeStorageServicer_to_server(
        MemeStorageService(app, config), server
    )
    server.add_insecure_port(config.port)
    server.start()

    print(f"Running grpc server on: {config.port}")
    return server
