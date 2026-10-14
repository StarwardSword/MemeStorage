import grpc
from concurrent import futures
from .proto import service_pb2_grpc
from .service import MemeStorageService
from app import Application

def serve(app: Application, port: str = "[::]:50051"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_MemeStorageServicer_to_server(MemeStorageService(app), server)
    server.add_insecure_port(port)
    server.start()
    print(f'Running grpc server on: {port}')
    server.wait_for_termination()