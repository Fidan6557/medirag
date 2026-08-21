"""Run the complete pipeline against the sample PDF and live Groq API."""

import asyncio

from src.pipeline import MediRAGPipeline


async def main() -> None:
    pipeline = MediRAGPipeline()
    pipeline.clear_knowledge_base()

    print("\nLoading and indexing the sample document...\n")
    await pipeline.ingest_document("data/raw/data_for_medirag.pdf")

    questions = [
        "What are essential medicines?",
        "How are medicines selected?",
        "What is the role of WHO in medicine selection?",
    ]

    print("\n" + "=" * 50)
    print("QUESTIONS")
    print("=" * 50)

    for question in questions:
        result = await pipeline.ask(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Score: {result['score']:.2f}")
        print(f"Time: {result['time']:.1f}s")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
