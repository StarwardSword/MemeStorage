import ctypes
import gc
import sys
from collections import OrderedDict
from threading import Lock, Semaphore, Timer
from typing import cast, override

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modality_types import MultimodalInput

from app import (
    EmbeddingProvider,
    Embeddings,
    FileDescriptor,
    FileStorage,
    UnsupportedTypeError,
)

from .configuration import Configuration

MODEL_NAME = "google/embeddinggemma-2"
# Known up front, so startup does not have to load the model to measure it
NATIVE_DIM = 768

QUERY = "SearchQuery"
DOCUMENT = "Document"


class Token:
    IMAGE = "<|image|>"
    AUDIO = "<|audio|>"
    VIDEO = "<|video|>"


class Key:
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


def mime_to_token(mime: str) -> str:
    if mime.startswith("image"):
        return Token.IMAGE
    if mime.startswith("audio"):
        return Token.AUDIO
    if mime.startswith("video"):
        return Token.VIDEO

    raise UnsupportedTypeError(mime)


def token_to_key(token: str) -> str:
    match token:
        case Token.IMAGE:
            return Key.IMAGE
        case Token.AUDIO:
            return Key.AUDIO
        case Token.VIDEO:
            return Key.VIDEO
    raise ValueError(token)


Query = dict[str, str]


# Freed torch memory stays in glibc arenas and is not returned to the OS by itself,
# without this the RSS barely drops after the model is unloaded
def release_memory_to_os() -> None:
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


class GemmaEmbeddings(EmbeddingProvider):
    def __init__(self, config: Configuration, file_storage: FileStorage):
        if config.threads is not None:
            torch.set_num_threads(config.threads)
            # Interop pool stays tiny, extra threads only fight for the same cores
            torch.set_num_interop_threads(1)

        self.config = config
        self.file_storage = file_storage
        self.semaphore = Semaphore(config.semaphore_count)
        self.max_batch = config.max_batch

        # Guards the model reference and the count of calls using it
        self._lock = Lock()
        self._model: SentenceTransformer | None = None
        self._users = 0
        self._timer: Timer | None = None

        self.dimension = config.truncate_dim or NATIVE_DIM

        # Search text -> embedding. Popular queries then skip the model, which may be unloaded
        self._query_cache: OrderedDict[str, Embeddings] = OrderedDict()
        self._query_cache_lock = Lock()

        # Load once now, so a download or login problem stops the start and not the first user.
        # With idle_unload_seconds the release below schedules the unload
        self._acquire()
        self._release()

    def _load(self) -> SentenceTransformer:
        return SentenceTransformer(
            MODEL_NAME,
            device=self.config.device,
            truncate_dim=self.config.truncate_dim,
            model_kwargs={"torch_dtype": self.config.dtype},
        )

    def _acquire(self) -> SentenceTransformer:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._model is None:
                print("Loading gemma embedding model")
                self._model = self._load()
            self._users += 1
            return self._model

    def _release(self) -> None:
        with self._lock:
            self._users -= 1
            idle = self.config.idle_unload_seconds
            if self._users > 0 or idle is None:
                return
            self._timer = Timer(idle, self._unload)
            self._timer.daemon = True
            self._timer.start()

    def _unload(self) -> None:
        with self._lock:
            # A request may have come in between the timer firing and the lock
            if self._users > 0 or self._model is None:
                return
            print("Unloading idle gemma embedding model")
            self._model = None
            self._timer = None
        release_memory_to_os()

    def _encode_single(self, query: Query, query_type: str) -> Embeddings:
        return self._encode_multiple([query], query_type)[0]

    def _encode_multiple(
        self, queries: list[Query], query_type: str
    ) -> list[Embeddings]:
        with self.semaphore:
            model = self._acquire()
            try:
                with torch.inference_mode():
                    arr = model.encode(
                        cast(MultimodalInput, queries),
                        prompt_name=query_type,
                        batch_size=min(max(1, len(queries)), self.max_batch),
                    )
            finally:
                self._release()

        arr = np.asarray(arr, dtype=np.float32)
        if self.config.truncate_dim is not None:
            # Matryoshka vectors are only meaningful after renormalizing the cut
            arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        res: list[Embeddings] = list(arr)
        return res

    def _document_descriptors(self, items: list[FileDescriptor]) -> list[Embeddings]:
        TOKEN, NAME = 0, 1
        tokens = [
            (mime_to_token(x.content_type), str(self.file_storage.get_full_path(x.id)))
            for x in items
        ]

        return self._encode_multiple(
            [
                {Key.TEXT: str(item[TOKEN]), token_to_key(item[TOKEN]): item[NAME]}
                for item in tokens
            ],
            DOCUMENT,
        )

    def _document_text(self, items: list[str]) -> list[Embeddings]:
        return self._encode_multiple([{Key.TEXT: x} for x in items], DOCUMENT)

    def _cached_query(self, text: str) -> Embeddings | None:
        with self._query_cache_lock:
            found = self._query_cache.get(text)
            if found is not None:
                self._query_cache.move_to_end(text)
            return found

    def _remember_query(self, text: str, embedding: Embeddings) -> None:
        with self._query_cache_lock:
            self._query_cache[text] = embedding
            self._query_cache.move_to_end(text)
            while len(self._query_cache) > self.config.query_cache_size:
                self._query_cache.popitem(last=False)

    @override
    def query(self, queries: list[str]) -> list[Embeddings]:
        if self.config.query_cache_size == 0:
            return self._encode_multiple(
                queries=[{Key.TEXT: x} for x in queries], query_type=QUERY
            )

        found = {x: self._cached_query(x) for x in dict.fromkeys(queries)}
        missing = [x for x, e in found.items() if e is None]
        if missing:
            encoded = self._encode_multiple(
                queries=[{Key.TEXT: x} for x in missing], query_type=QUERY
            )
            for text, embedding in zip(missing, encoded):
                found[text] = embedding
                self._remember_query(text, embedding)

        return [cast(Embeddings, found[x]) for x in queries]

    @override
    def document(self, items: list[FileDescriptor] | list[str]) -> list[Embeddings]:
        if len(items) == 0:
            return []
        if isinstance(items[0], str):
            return self._document_text(cast(list[str], items))
        return self._document_descriptors(cast(list[FileDescriptor], items))

    @override
    def validate_embedding_dimension(self, embedding: Embeddings) -> bool:
        return len(embedding) == self.dimension
