"""Pure formatting helpers used by the Gradio presentation layer."""

import re
from typing import Any

_MARKDOWN_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|])")


def escape_markdown(value: Any) -> str:
    """Escape Markdown control characters in untrusted display values."""
    return _MARKDOWN_SPECIAL_RE.sub(r"\\\1", str(value))


def format_processing_status(results: list[dict], total_chunks: int) -> str:
    """Build a concise upload status message from pipeline results."""
    successful = [result for result in results if result["success"]]
    failed = [result for result in results if not result["success"]]

    if successful:
        lines = [
            "### ✅ Knowledge base ready",
            f"**{len(successful)} document(s)** · **{total_chunks} searchable chunks**",
        ]
        lines.extend(
            f"- `{escape_markdown(result['file'])}` — {result['chunks']} chunks"
            for result in successful
        )
        if failed:
            lines.append("\n**Skipped files**")
    else:
        lines = [
            "### ⚠️ Nothing was indexed",
            "The existing knowledge base was kept unchanged.",
        ]

    lines.extend(
        f"- `{escape_markdown(result['file'])}` — "
        f"{escape_markdown(result.get('error', 'processing failed'))}"
        for result in failed
    )
    return "\n".join(lines)


def format_index_status(total_chunks: int) -> str:
    """Format the persisted index state shown when the interface opens."""
    if total_chunks <= 0:
        return "### Knowledge base is empty\nUpload documents to begin."
    return (
        "### ✅ Existing knowledge base ready\n"
        f"**{total_chunks} searchable chunks** loaded from the local index."
    )
