"""
test_day3.py — Full Pipeline Test
PDF → loader → chunker → embedder → vectorstore → retriever → generator
"""

import asyncio
from src.pipeline import MediRAGPipeline


async def main():
    # 1) Pipeline yarat
    pipeline = MediRAGPipeline()

    # 2) Köhnə data-nı təmizlə
    pipeline.clear_knowledge_base()

    # 3) PDF-i yüklə və işlə
    print("\nDocument is loading...\n")
    await pipeline.ingest_document("data/raw/data_for_medirag.pdf")

    # 4) Suallar ver
    questions = [
        "What are essential medicines?",
        "How are medicines selected?",
        "What is the role of WHO in medicine selection?",
    ]

    print("\n" + "="*50)
    print("QUESTIONS")
    print("="*50)

    for q in questions:
        result = await pipeline.ask(q)
        print(f"\nQuestion: {q}")
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Score: {result['score']:.2f}")
        print(f"Time: {result['time']:.1f}s")
        print("-"*50)


if __name__ == "__main__":
    asyncio.run(main())
