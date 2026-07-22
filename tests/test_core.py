import tempfile
import unittest
from pathlib import Path

from src.chunker import chunk_text
from src.generator import Generator
from src.loader import load_document
from src.retriever import Retriever, _lexical_overlap


class FakeEmbedder:
    def embed_text(self, text):
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def search(self, query_embedding, top_k):
        return self.results[:top_k]


class LoaderAndChunkerTests(unittest.TestCase):
    def test_load_text_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("Paracetamol reduces fever.", encoding="utf-8")

            pages = load_document(str(path))

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["metadata"]["source"], "notes.txt")
        self.assertEqual(pages[0]["metadata"]["format"], "txt")

    def test_empty_text_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text("   ", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty"):
                load_document(str(path))

    def test_unsupported_extension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported format"):
            load_document("document.csv")

    def test_chunk_contains_original_metadata(self):
        chunks = chunk_text(
            "A short medical note.",
            {"source": "note.txt", "page": 1, "format": "txt"},
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["source"], "note.txt")
        self.assertEqual(chunks[0]["metadata"]["chunk_index"], 0)


class RetrieverTests(unittest.TestCase):
    def test_lexical_overlap_ignores_common_words(self):
        score = _lexical_overlap(
            "What is paracetamol used for?",
            "Paracetamol is used to reduce fever.",
        )

        self.assertEqual(score, 1.0)

    def test_weak_matches_are_not_answerable(self):
        store = FakeVectorStore([
            {
                "text": "A completely unrelated passage.",
                "metadata": {"source": "note.txt", "page": 1},
                "score": 0.1,
            }
        ])
        result = Retriever(FakeEmbedder(), store).retrieve(
            "paracetamol dosage",
            threshold=0.3,
        )

        self.assertFalse(result["is_answerable"])
        self.assertEqual(result["results"], [])

    def test_exact_keyword_match_is_retained(self):
        store = FakeVectorStore([
            {
                "text": "Paracetamol can reduce fever.",
                "metadata": {"source": "note.txt", "page": 2},
                "score": 0.5,
            }
        ])
        result = Retriever(FakeEmbedder(), store).retrieve(
            "paracetamol fever",
            threshold=0.3,
        )

        self.assertTrue(result["is_answerable"])
        self.assertEqual(result["results"][0]["metadata"]["page"], 2)


class GeneratorTests(unittest.TestCase):
    def test_unanswerable_result_does_not_require_api_key(self):
        result = Generator().generate(
            "Unknown question",
            "",
            {"is_answerable": False, "best_score": 0.12, "results": []},
        )

        self.assertFalse(result["answered"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["score"], 0.12)


if __name__ == "__main__":
    unittest.main()
