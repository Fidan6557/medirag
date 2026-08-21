import unittest

from src.vectorstore import VectorStore


class FakeCollection:
    def __init__(self):
        self.deleted_sources = []
        self.upsert_payload = None

    def delete(self, where):
        self.deleted_sources.append(where["source"])

    def upsert(self, **payload):
        self.upsert_payload = payload

    def count(self):
        return len(self.upsert_payload["ids"]) if self.upsert_payload else 0


class VectorStoreTests(unittest.TestCase):
    def test_same_named_documents_use_distinct_content_ids(self):
        store = VectorStore.__new__(VectorStore)
        store.collection = FakeCollection()
        chunks = [
            {
                "text": "First document",
                "embedding": [1.0, 0.0],
                "metadata": {
                    "source": "report.pdf",
                    "document_id": "first",
                    "page": 1,
                    "chunk_index": 0,
                },
            },
            {
                "text": "Second document",
                "embedding": [0.0, 1.0],
                "metadata": {
                    "source": "report.pdf",
                    "document_id": "second",
                    "page": 1,
                    "chunk_index": 0,
                },
            },
        ]

        store.add_chunks(chunks)

        self.assertEqual(store.collection.deleted_sources, ["report.pdf"])
        self.assertEqual(
            store.collection.upsert_payload["ids"],
            ["first_page_1_chunk_0", "second_page_1_chunk_0"],
        )


if __name__ == "__main__":
    unittest.main()
