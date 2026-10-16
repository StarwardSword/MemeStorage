import grpc
from concurrent import futures
from .proto import service_pb2_grpc
from .service import MemeStorageService
from .config import Configuration
from app import Application

def serve(app: Application, config: Configuration):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))
    service_pb2_grpc.add_MemeStorageServicer_to_server(MemeStorageService(app), server)
    server.add_insecure_port(config.port)
    server.start()

    print(f'Running grpc server on: {config.port}')
    server.wait_for_termination()