import uuid
import chromadb
import numpy as np
from .config import Configuration
from app import Storage, Record, RecordNotFoundError, Embeddings

class ChromaStorage(Storage):
    def __init__(self, config: Configuration) -> None:
        self.client = chromadb.PersistentClient(path=config.path)
        self.primary_collection = self.client.get_or_create_collection("memes_primary")
        self.text_description_collection = self.client.get_or_create_collection("memes_text_description")

    def save(self, records: list[Record]) -> None:
        if any([x for x in records if x.embedding is None]): raise ValueError(f"ChromaStorage.save got record with primary embeddings None")

        ids = [str(r.id) for r in records]
        primary_embeddings = [r.embedding for r in records]
        documents = [
            r.text_description if r.text_description is not None else "" 
            for r in records
        ]

        self.primary_collection.add(ids=ids, embeddings=np.asarray(primary_embeddings), documents=documents)

        # WARN: If something goes wrong THIS thing is bugged
        non_null_secondary_idx = set([i for (i, x) in enumerate(records) if x.text_description is not None])        
        if not non_null_secondary_idx: return

        non_null_secondaries = [x for (i, x) in enumerate(records) if i in non_null_secondary_idx]
        ids = [x for (i, x) in enumerate(ids) if i in non_null_secondary_idx]
        secondary_embeddings = [x.text_embedding for x in non_null_secondaries]
        documents = [x for (i, x) in enumerate(documents) if i in non_null_secondary_idx]

        self.text_description_collection.add(ids=ids, embeddings=np.asarray(secondary_embeddings), documents=documents)

    def get_by_id(self, ids: list[uuid.UUID]) -> list[Record]: 
        strids = [str(x) for x in ids]
        res = self.primary_collection.get(ids=strids, include=["documents"])

        if res["documents"] is None: raise Exception("chroma: can not load documents")

        items = {
            uuid.UUID(id_): doc
            for id_, doc in zip(res["ids"], res["documents"])
        }

        missing = [i for i in ids if i not in items]
        if missing: raise RecordNotFoundError(missing)

        return [
            Record(
                id=id,
                embedding=None,
                text_description=items[id],
                text_embedding=None
            ) for id in ids
        ]

    def get(self, embeddings: list[Embeddings], n_records: int) -> list[list[Record]]:
        primary_results = self._get_from_single_collection(embeddings, n_records, self.primary_collection)
        secondary_results = self._get_from_single_collection(embeddings, n_records, self.text_description_collection)

        RECORD, DISTANCE = 0, 1

        # It has to return at least empty array for each query
        assert(len(primary_results) == len(secondary_results))
        res: list[list[Record]] = []

        for i in range(len(primary_results)):
            items = {x[RECORD].id: (x[RECORD], x[DISTANCE]) for x in primary_results[i]} 

            for secondary in secondary_results[i]:
                # if items does not contain the thing OR items contains the thing but its distance is higher than secondary
                if secondary[RECORD].id not in items or secondary[DISTANCE] < items[secondary[RECORD].id][DISTANCE]:
                    items[secondary[RECORD].id] = secondary

            # Secondary-only matches are appended after primary ones, so the merged list has to be sorted again
            closest = sorted(items.values(), key=lambda x: x[DISTANCE])
            records = [x[RECORD] for x in closest[:n_records]]
            res.append(records)

        return res

    # For each embedding returns list of n closest items with their distance
    def _get_from_single_collection(self, embeddings: list[Embeddings], n_records: int, collection: chromadb.Collection) -> list[list[tuple[Record, float]]]:
        query = collection.query(np.asarray(embeddings), n_results=n_records, include=["documents", "distances"])
        if query["documents"] is None: raise Exception("chroma: can not load documents")
        if query["distances"] is None: raise Exception("chroma: can not load distances")

        ID, DIST, DOC = 0, 1, 2
        items = [
            (id_, distance, doc) 
            for id_, doc, distance in zip(query["ids"], query["documents"], query["distances"])
        ]

        #                       Record, Dist
        records: list[list[tuple[Record, float]]] = []
        for embedding_results in items:
            batch: list[tuple[Record, float]] = [(
                Record(
                    id=uuid.UUID(embedding_results[ID][idx]),
                    embedding=None,
                    text_description=embedding_results[DOC][idx],
                    text_embedding=None
                ), embedding_results[DIST][idx]
                ) for idx in range(len(embedding_results[ID]))
            ]

            records.append(batch)

        return records

    def update(self, records: list[Record]): 
        raise NotImplementedError

    def remove(self, records: list[uuid.UUID] | list[Record]): 
        ids = [str(r) if isinstance(r, uuid.UUID) else str(r.id) for r in records ]
        self.primary_collection.delete(ids=ids)
        self.text_description_collection.delete(ids=ids)