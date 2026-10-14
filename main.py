import torch
from api import serve_grpc
from pathlib import Path
from infrastructure import ChromaStorage, LocalFileStorage, GemmaEmbeddings
from app import Application

print("Initializing file storage")
files = LocalFileStorage(
    root=Path("./.runtime/files/"), 
    sqlite_connection_string="./.runtime/sqlite.db"
)

print("Initializing gemma embedding model")
embeddings = GemmaEmbeddings(
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    device="cuda" if torch.cuda.is_available() else "cpu",
    file_storage=files
)

print("Initializing chromadb")
db = ChromaStorage(
    Path("./.runtime/storage.chromadb"),
)

app = Application(embedding_provider=embeddings, storage=db, file_storage=files)

print("Starting grpc")
serve_grpc(app)