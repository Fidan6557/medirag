"""
generator.py — LLM Generation Module

Sends retriever context + user question to the Groq LLM.
Answers are always grounded in documents — hallucination is minimised.

Prompt engineering:
    - System prompt : defines the LLM's role
    - User prompt   : context + question
    - Temperature   : 0.1 — deterministic (important for medical context)

Error handling:
    - Rate-limit errors  : retried with exponential back-off (up to 3 attempts)
    - Auth errors        : surfaced immediately with a clear message
    - All other errors   : caught and returned as a structured failure dict
"""

import logging
import time
from typing import Dict, Optional

from groq import Groq, APIError, RateLimitError, AuthenticationError
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_MAX_RETRIES   = 3
_RETRY_DELAY_S = 2   # seconds; doubled on each retry

_QUERY_REWRITE_HINTS = (
    " nedir", " nədir", " ucun", " üçün", "istifade", "istifadə",
    "olunur", "derman", "dərman",
)


class Generator:
    """Generates answers using the Groq LLM."""

    def __init__(self):
        # Keep imports, document ingestion, and local retrieval usable without
        # an API key. A clear error is raised only when an LLM call is needed.
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.model  = GROQ_MODEL

        self.system_prompt = (
            "You are MediRAG, an intelligent medical document assistant.\n\n"
            "Your role:\n"
            "- Answer questions ONLY based on the provided document context.\n"
            "- If the answer is not in the context, say exactly: "
            "\"This information is not available in the provided documents.\"\n"
            "- Always cite your sources by mentioning the document name and page number.\n"
            "- Be precise and concise.\n"
            "- Do not make assumptions beyond what the documents state.\n"
            "- If a medical term has no clear translation, keep the standard "
            "medical term instead of inventing a literal translation.\n"
            "- Support English, Azerbaijani, and Russian — respond in the same "
            "language as the question.\n\n"
            "Remember: Patient safety depends on accuracy. Never guess."
        )

    # ── public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: str,
        retrieval_result: Dict,
        response_instruction: Optional[str] = None,
    ) -> Dict:
        """
        Generates an answer for *query* given *context*.

        Args:
            query            : the user's question
            context          : formatted context string from the retriever
            retrieval_result : output of Retriever.retrieve()

        Returns:
            {
                "answer"   : str,
                "sources"  : list[dict],
                "score"    : float,
                "answered" : bool,
            }
        """
        if not retrieval_result["is_answerable"]:
            return {
                "answer":   "This information is not available in the provided documents.",
                "sources":  [],
                "score":    retrieval_result["best_score"],
                "answered": False,
            }

        language_instruction = (
            response_instruction.strip()
            if response_instruction
            else "Respond in the same language as the question."
        )

        user_prompt = (
            "Based on the following medical document excerpts, answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"RESPONSE LANGUAGE: {language_instruction}\n\n"
            "Provide a clear, accurate answer based solely on the context above. "
            "At the end, mention which source(s) you used."
        )

        answer = self._call_with_retry(user_prompt)

        sources = [
            {
                "source": r["metadata"]["source"],
                "page":   r["metadata"]["page"],
                "score":  r["score"],
            }
            for r in retrieval_result["results"]
        ]

        return {
            "answer":   answer,
            "sources":  sources,
            "score":    retrieval_result["best_score"],
            "answered": True,
        }

    def should_rewrite_query_for_retrieval(self, query: str) -> bool:
        lowered = f" {query.lower()} "
        return any(ord(ch) > 127 for ch in query) or any(
            hint in lowered for hint in _QUERY_REWRITE_HINTS
        )

    def rewrite_query_for_retrieval(self, query: str) -> str:
        """
        Converts non-English or transliterated user questions into a short
        English search query for retrieval. Falls back to the original query.
        """
        prompt = (
            "Rewrite the user question as a short English search query for a "
            "medical document retrieval system. Preserve the meaning exactly. "
            "Normalize generic medicine names to standard English/WHO spellings "
            "when obvious. In Azerbaijani, 'vacib dərmanlar' means "
            "'essential medicines', not vaccines.\n\n"
            "Examples:\n"
            "vacib dərmanlar nədir? -> What are essential medicines?\n"
            "amoksisillin nə üçün istifadə olunur? -> amoxicillin uses\n\n"
            "Return only the rewritten query.\n\n"
            f"USER QUESTION: {query}"
        )

        try:
            client = self._require_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You rewrite queries for document search.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=80,
            )
            rewritten = response.choices[0].message.content.strip().strip('"')
            return rewritten or query
        except Exception as e:
            logger.warning(f"Query rewrite failed; using original query: {e}")
            return query

    # ── private ───────────────────────────────────────────────────────────────

    def _require_client(self) -> Groq:
        if self.client is None:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. Copy .env.example to .env "
                "and add your key from https://console.groq.com."
            )
        return self.client

    def _call_with_retry(self, user_prompt: str) -> str:
        """
        Calls the Groq API with exponential back-off on rate-limit errors.
        Raises a RuntimeError with a user-friendly message on permanent failure.
        """
        delay = _RETRY_DELAY_S
        client = self._require_client()

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                return response.choices[0].message.content

            except AuthenticationError:
                raise RuntimeError(
                    "Groq authentication failed. "
                    "Please check your GROQ_API_KEY in the .env file."
                )

            except RateLimitError:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        "Groq rate limit reached. Please wait a moment and try again."
                    )
                logger.warning(
                    f"Groq rate limit hit (attempt {attempt}/{_MAX_RETRIES}). "
                    f"Retrying in {delay}s…"
                )
                time.sleep(delay)
                delay *= 2

            except APIError as e:
                logger.error(f"Groq API error: {e}")
                raise RuntimeError(f"LLM service error: {e}") from e

            except Exception as e:
                logger.error(f"Unexpected error calling Groq: {e}")
                raise RuntimeError(f"Unexpected error: {e}") from e

        # Should never reach here
        raise RuntimeError("LLM generation failed after maximum retries.")
