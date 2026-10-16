from api import serve_grpc
from infrastructure import ChromaStorage, LocalFileStorage, GemmaEmbeddings
from app import Application
from configure import Configuration

config = Configuration()

print("Initializing file storage")
files = LocalFileStorage(config.files)

print("Initializing gemma embedding model")
embeddings = GemmaEmbeddings(
    config=config.gemma,
    file_storage=files
)

print("Initializing chromadb")
db = ChromaStorage(config.chroma)

app = Application(embedding_provider=embeddings, storage=db, file_storage=files)

if config.grpc is not None:
    print("Starting grpc")
    serve_grpc(app, config.grpc)
else:
    print("Nothing started: no api specified in config")