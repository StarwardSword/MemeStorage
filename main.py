import os
from pathlib import Path

# torchcodec looks for FFmpeg DLLs once, when sentence_transformers is imported,
# so the directory from `make ffmpeg` is added before importing infrastructure.
# Without it torchcodec falls back to ffmpeg from PATH
FFMPEG_DIR = Path(__file__).parent / "ffmpeg"
if hasattr(os, "add_dll_directory") and FFMPEG_DIR.is_dir():
    ffmpeg_dlls = os.add_dll_directory(str(FFMPEG_DIR))

from api import serve_grpc, serve_http
from app import Application
from configure import Configuration
from infrastructure import ChromaStorage, GemmaEmbeddings, LocalFileStorage

GRPC_STOP_GRACE = 5

config = Configuration()

print("Initializing file storage")
files = LocalFileStorage(config.files)

print("Initializing gemma embedding model")
embeddings = GemmaEmbeddings(config=config.gemma, file_storage=files)

print("Initializing chromadb")
db = ChromaStorage(config.chroma)

app = Application(
    embedding_provider=embeddings,
    storage=db,
    file_storage=files,
    configuration=config.app,
)

grpc_server = None
if config.grpc is not None:
    print("Starting grpc")
    grpc_server = serve_grpc(app, config.grpc)

# gRPC serves from its own threads, so the main thread is given to http or just waits
try:
    if config.http is not None:
        print("Starting http")
        serve_http(app, config.http)
    elif grpc_server is not None:
        grpc_server.wait_for_termination()
    else:
        print("Nothing started: no api specified in config")
except KeyboardInterrupt:
    pass
finally:
    if grpc_server is not None:
        print("Stopping grpc")
        grpc_server.stop(GRPC_STOP_GRACE).wait()
