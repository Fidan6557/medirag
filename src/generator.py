"""
generator.py — LLM Generation Module

Sends the context from the retriever + the question to the Groq LLM.
The answer is always based on the document — hallucination is minimized.

Prompt engineering:
    - System prompt: defines the role of the LLM
    - User prompt: context + question
    - Temperature: 0.1 — deterministic, important for medical context
"""

from groq import Groq
from typing import Dict
from config import GROQ_API_KEY, GROQ_MODEL


class Generator:
    """
    Generator class — generates answers using the Groq LLM.
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

        # System prompt — defines how the LLM should behave
        self.system_prompt = """You are MediRAG, an intelligent medical document assistant.

Your role:
- Answer questions ONLY based on the provided document context
- If the answer is not in the context, say: "This information is not available in the provided documents."
- Always cite your sources by mentioning the document name and page number
- Be precise and concise
- Do not make assumptions beyond what the documents state
- Support English, Azerbaijani, and Russian — respond in the same language as the question

Remember: Patient safety depends on accuracy. Never guess."""

    def generate(self, query: str, context: str, retrieval_result: Dict) -> Dict:
        """
        Generates an answer based on the question + context.

        Args:
            query            : the user's question
            context          : formatted context from the retriever
            retrieval_result : output of retrieve() (is_answerable, best_score)

        Returns:
            {
                "answer"    : "...",
                "sources"   : [...],
                "score"     : 0.87,
                "answered"  : True
            }
        """
        # Out-of-scope detection — check before sending to the LLM
        if not retrieval_result["is_answerable"]:
            return {
                "answer": "This information is not available in the provided documents.",
                "sources": [],
                "score": retrieval_result["best_score"],
                "answered": False
            }

        # User prompt — context + question
        user_prompt = f"""Based on the following medical document excerpts, answer the question.

CONTEXT:
{context}

QUESTION: {query}

Provide a clear, accurate answer based solely on the context above.
At the end, mention which source(s) you used."""

        # Groq API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.1,   # Aşağı temperature = daha deterministik cavab
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        # Extract sources
        sources = [
            {
                "source": r["metadata"]["source"],
                "page"  : r["metadata"]["page"],
                "score" : r["score"]
            }
            for r in retrieval_result["results"]
        ]

        return {
            "answer" : answer,
            "sources": sources,
            "score"  : retrieval_result["best_score"],
            "answered": True
        }