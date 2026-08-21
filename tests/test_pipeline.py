import asyncio
import tempfile
import unittest
from pathlib import Path

from src.pipeline import MediRAGPipeline


class FakeEmbedder:
    def __init__(self):
        self.last_text = None

    def embed_chunks(self, chunks):
        return [{**chunk, "embedding": [1.0, 0.0]} for chunk in chunks]

    def embed_text(self, text):
        self.last_text = text
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

    def clear(self):
        self.chunks = []


class FakeGenerator:
    def should_rewrite_query_for_retrieval(self, query):
        return False


class RewritingGenerator(FakeGenerator):
    def should_rewrite_query_for_retrieval(self, query):
        return True

    def rewrite_query_for_retrieval(self, query):
        return "essential medicines"

    def generate(
        self,
        query,
        context,
        retrieval_result,
        response_instruction=None,
    ):
        return {
            "answer": "Grounded answer.",
            "sources": [],
            "score": retrieval_result["best_score"],
            "answered": retrieval_result["is_answerable"],
        }


class SearchableVectorStore(FakeVectorStore):
    def __init__(self):
        super().__init__()
        self.chunks = [
            {
                "text": "Essential medicines meet priority healthcare needs.",
                "metadata": {"source": "guide.pdf", "page": 2},
                "score": 0.8,
            }
        ]

    def search(self, query_embedding, top_k=3):
        return self.chunks[:top_k]


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

    def test_empty_knowledge_base_does_not_load_embedding_model(self):
        pipeline = MediRAGPipeline(
            vectorstore=FakeVectorStore(),
            generator=FakeGenerator(),
        )

        result = asyncio.run(pipeline.ask("What is this?"))

        self.assertFalse(result["answered"])
        self.assertIsNone(pipeline.embedder)

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

    def test_failed_replacement_keeps_existing_knowledge_base(self):
        pipeline = self.make_pipeline()
        pipeline.vectorstore.chunks = [{"text": "Existing indexed content."}]

        results = asyncio.run(
            pipeline.replace_documents(["definitely-missing-document.txt"])
        )

        self.assertFalse(results[0]["success"])
        self.assertEqual(pipeline.vectorstore.count(), 1)
        self.assertEqual(
            pipeline.vectorstore.chunks[0]["text"],
            "Existing indexed content.",
        )

    def test_successful_replacement_removes_old_content(self):
        pipeline = self.make_pipeline()
        pipeline.vectorstore.chunks = [{"text": "Old indexed content."}]

        with tempfile.TemporaryDirectory() as directory:
            replacement = Path(directory) / "replacement.txt"
            replacement.write_text("New clinical guidance.", encoding="utf-8")
            results = asyncio.run(pipeline.replace_documents([str(replacement)]))

        self.assertTrue(results[0]["success"])
        self.assertEqual(pipeline.vectorstore.count(), 1)
        self.assertEqual(
            pipeline.vectorstore.chunks[0]["text"],
            "New clinical guidance.",
        )
        self.assertIn(
            "document_id",
            pipeline.vectorstore.chunks[0]["metadata"],
        )

    def test_query_rewrite_replaces_original_retrieval_query(self):
        embedder = FakeEmbedder()
        pipeline = MediRAGPipeline(
            embedder=embedder,
            vectorstore=SearchableVectorStore(),
            generator=RewritingGenerator(),
        )

        result = asyncio.run(pipeline.ask("Vacib dərmanlar nədir?"))

        self.assertTrue(result["answered"])
        self.assertEqual(embedder.last_text, "essential medicines")


if __name__ == "__main__":
    unittest.main()
