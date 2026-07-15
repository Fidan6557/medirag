import asyncio
import tempfile
import unittest
from pathlib import Path

from src.pipeline import MediRAGPipeline


class FakeEmbedder:
    def embed_chunks(self, chunks):
        return [{**chunk, "embedding": [1.0, 0.0]} for chunk in chunks]

    def embed_text(self, text):
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self):
        self.chunks = []

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)

    def count(self):
        return len(self.chunks)

    def search(self, query_embedding, top_k=3):
        return []


class FakeGenerator:
    def should_rewrite_query_for_retrieval(self, query):
        return False


class PipelineTests(unittest.TestCase):
    def make_pipeline(self):
        return MediRAGPipeline(
            embedder=FakeEmbedder(),
            vectorstore=FakeVectorStore(),
            generator=FakeGenerator(),
        )

    def test_empty_knowledge_base_returns_without_llm_call(self):
        result = asyncio.run(self.make_pipeline().ask("What is this?"))

        self.assertFalse(result["answered"])
        self.assertEqual(result["sources"], [])
        self.assertIn("No documents", result["answer"])

    def test_batch_ingestion_keeps_processing_after_bad_file(self):
        pipeline = self.make_pipeline()
        with tempfile.TemporaryDirectory() as directory:
            valid_file = Path(directory) / "notes.txt"
            valid_file.write_text("Paracetamol reduces fever.", encoding="utf-8")
            missing_file = Path(directory) / "missing.txt"

            results = asyncio.run(
                pipeline.ingest_many([str(valid_file), str(missing_file)])
            )

        self.assertEqual([item["success"] for item in results], [True, False])
        self.assertEqual(pipeline.vectorstore.count(), 1)
        self.assertIn("error", results[1])

    def test_blank_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            asyncio.run(self.make_pipeline().ask("   "))


if __name__ == "__main__":
    unittest.main()
