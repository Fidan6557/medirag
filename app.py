"""
app.py — MediRAG Gradio UI
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Optional

import gradio as gr

from src.pipeline import MediRAGPipeline

logger  = logging.getLogger(__name__)
_pipeline: Optional[MediRAGPipeline] = None
_pipeline_lock = Lock()


def get_pipeline() -> MediRAGPipeline:
    """Initialise heavy ML dependencies only when the app first needs them."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = MediRAGPipeline()
    return _pipeline


# ── callbacks ─────────────────────────────────────────────────────────────────

async def process_documents(files):
    if not files:
        return "⚠️ No files uploaded."

    pipeline = get_pipeline()
    pipeline.clear_knowledge_base()

    names = []
    errors = []
    results = await pipeline.ingest_many([file.name for file in files])
    for result in results:
        if result["success"]:
            names.append(result["file"])
        else:
            errors.append(
                f"✗ {result['file']}: {result.get('error', 'processing failed')}"
            )

    stats = pipeline.get_stats()
    lines = [f"✅ {len(names)} document(s) processed — {stats['total_chunks']} chunks indexed."]
    if errors:
        lines.append("\n**Errors:**\n" + "\n".join(errors))
    return "\n".join(lines)


async def chat(message: str, history: list, language: str):
    """Answer one question using the currently indexed documents."""
    lang_hint = {
        "English":    "Respond in English.",
        "Azərbaycan": "Cavabı Azərbaycan dilində ver.",
        "Русский":    "Отвечай на русском языке.",
    }.get(language, "Respond in English.")

    try:
        pipeline = get_pipeline()
        result = await pipeline.ask(message, response_instruction=lang_hint)
    except RuntimeError as e:
        # Propagated from Generator (auth error, rate limit, etc.)
        return f"⚠️ **Service error:** {e}"
    except Exception as e:
        logger.exception("Unexpected error during pipeline.ask()")
        return f"⚠️ **Unexpected error:** {e}"

    if not result["answered"]:
        return f"❌ {result['answer']}"

    sources = ""
    if result.get("sources"):
        sources = "\n\n**Sources:**\n" + "\n".join(
            f"- 📄 {s['source']} — Page {s['page']} (score: {s['score']:.2f})"
            for s in result["sources"]
        )

    score = result["score"]
    relevance = (
        "🟢 High" if score >= 0.55
        else "🟡 Medium" if score >= 0.30
        else "🔴 Low"
    )

    return (
        f"{result['answer']}{sources}\n\n"
        f"`Retrieval relevance: {relevance} ({score:.0%}) · "
        f"⏱ {result['time']:.1f}s`"
    )


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background-color: #F0FDF4 !important; }
"""

with gr.Blocks(title="MediRAG") as demo:

    gr.Markdown("""
    # 🩺 MediRAG
    **Medical Document Intelligence System** — Ask questions about your medical documents.

    > **Medical disclaimer:** Answers may be incomplete or incorrect. Do not use
    > MediRAG as a substitute for diagnosis, treatment, or professional medical advice.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Documents")
            files_input = gr.File(
                label="Upload Documents",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                file_count="multiple",
            )
            language = gr.Dropdown(
                choices=["English", "Azərbaycan", "Русский"],
                value="English",
                label="🌍 Language",
            )
            process_btn = gr.Button("⚙️ Process Documents", variant="primary")
            status = gr.Textbox(label="Status", interactive=False, lines=3)

            process_btn.click(
                fn=process_documents,
                inputs=[files_input],
                outputs=[status],
            )

        with gr.Column(scale=3):
            gr.Markdown("### 💬 Chat")
            gr.ChatInterface(
                fn=chat,
                additional_inputs=[language],
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    scale=7,
                ),
            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        css=CSS,
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="stone"),
    )
