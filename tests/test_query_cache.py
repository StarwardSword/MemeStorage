import unittest
from collections import OrderedDict
from threading import Lock

import numpy as np

from infrastructure.embedding_providers.configuration import Configuration
from infrastructure.embedding_providers.gemma import GemmaEmbeddings


def provider(cache_size: int) -> GemmaEmbeddings:
    # __init__ would load the model, the cache does not need it
    g = object.__new__(GemmaEmbeddings)
    g.config = Configuration(max_batch=1, query_cache_size=cache_size)
    g._query_cache = OrderedDict()
    g._query_cache_lock = Lock()
    g.encoded = []

    def encode(queries, query_type):
        g.encoded.append([q["text"] for q in queries])
        return [np.full(2, ord(q["text"]), dtype=np.float32) for q in queries]

    g._encode_multiple = encode
    return g


class QueryCacheTest(unittest.TestCase):
    def test_duplicates_are_encoded_once_and_keep_order(self):
        g = provider(8)
        res = g.query(["a", "b", "a"])
        self.assertEqual([int(x[0]) for x in res], [97, 98, 97])
        self.assertEqual(g.encoded, [["a", "b"]])
        g.query(["b"])
        self.assertEqual(len(g.encoded), 1)

    def test_oldest_is_evicted(self):
        g = provider(1)
        g.query(["a"])
        g.query(["b"])
        g.query(["a"])
        self.assertEqual(g.encoded, [["a"], ["b"], ["a"]])

    def test_size_zero_never_caches(self):
        g = provider(0)
        g.query(["a"])
        g.query(["a"])
        self.assertEqual(len(g.encoded), 2)
        self.assertEqual(len(g._query_cache), 0)

    def test_truncate_dim_rejects_odd_sizes(self):
        with self.assertRaises(ValueError):
            Configuration(max_batch=1, truncate_dim=1024)


if __name__ == "__main__":
    unittest.main()
