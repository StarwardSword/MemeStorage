from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modality_types import MultimodalInput
from torch import dtype
from app import EmbeddingProvider, Embeddings, FileDescriptor, FileStorage
from typing import cast, override

MODEL_NAME = "google/embeddinggemma-2"

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
    if mime.startswith('image'): return Token.IMAGE
    if mime.startswith('audio'): return Token.AUDIO
    if mime.startswith('video'): return Token.VIDEO
    
    raise ValueError(mime)

def token_to_key(token: str) -> str:
    match token:
        case Token.IMAGE: return Key.IMAGE
        case Token.AUDIO: return Key.AUDIO
        case Token.VIDEO: return Key.VIDEO
    raise ValueError(token)

Query = dict[str, str]

class GemmaEmbeddings(EmbeddingProvider):
    def __init__(self, dtype: dtype, device: str, file_storage: FileStorage):
        self.model = SentenceTransformer(MODEL_NAME, device=device, model_kwargs={"torch_dtype": dtype})
        self.file_storage = file_storage

    def _encode_single(self, query: Query, query_type: str) -> Embeddings:
        return self._encode_multiple([query], query_type)[0]

    def _encode_multiple(self, queries: list[Query], query_type: str) -> list[Embeddings]:
        # WARN: slow af
        arr = self.model.encode(cast(MultimodalInput, queries), prompt_name=query_type)
        res: list[Embeddings] = list(arr)

        return res

    def _document_descriptors(self, items: list[FileDescriptor]) -> list[Embeddings]: 
        TOKEN, NAME = 0, 1
        tokens = [(mime_to_token(x.content_type), str(self.file_storage.get_full_path(x.id))) for x in items]

        return self._encode_multiple([{
            Key.TEXT: str(item[TOKEN]),
            token_to_key(item[TOKEN]): item[NAME]
        } for item in tokens], DOCUMENT)

    def _document_text(self, items: list[str]) -> list[Embeddings]: 
        return self._encode_multiple([{
            Key.TEXT: x
        } for x in items], DOCUMENT)

    @override
    def query(self, queries: list[str]) -> list[Embeddings]: 
        return self._encode_multiple(
            queries=[{
                Key.TEXT: x
            } for x in queries],
            query_type=QUERY
        )
    
    @override
    def document(self, items: list[FileDescriptor] | list[str]) -> list[Embeddings]: 
        if len(items) == 0: return []
        if isinstance(items[0], str): return self._document_text(cast(list[str], items))
        return self._document_descriptors(cast(list[FileDescriptor], items))