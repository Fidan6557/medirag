"""
app.py — MediRAG Gradio UI
"""

import asyncio
import logging
from pathlib import Path

import gradio as gr

from src.pipeline import MediRAGPipeline

logger  = logging.getLogger(__name__)
pipeline = MediRAGPipeline()


# ── helpers ───────────────────────────────────────────────────────────────────

def run_async(coro):
    """Runs an async coroutine from synchronous Gradio callbacks."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


# ── callbacks ─────────────────────────────────────────────────────────────────

def process_documents(files):
    if not files:
        return "⚠️ No files uploaded."

    pipeline.clear_knowledge_base()

    names = []
    errors = []
    for file in files:
        try:
            run_async(pipeline.ingest_document(file.name))
            names.append(Path(file.name).name)
        except Exception as e:
            errors.append(f"✗ {Path(file.name).name}: {e}")

    stats = pipeline.get_stats()
    lines = [f"✅ {len(names)} document(s) processed — {stats['total_chunks']} chunks indexed."]
    if errors:
        lines.append("\n**Errors:**\n" + "\n".join(errors))
    return "\n".join(lines)


def chat(message: str, history: list, language: str):
    """
    Gradio ChatInterface callback.

    history is a list of [user_msg, assistant_msg] pairs — we include it
    in the context so the LLM can refer to earlier turns.
    """
    lang_hint = {
        "English":    "Respond in English.",
        "Azərbaycan": "Cavabı Azərbaycan dilində ver.",
        "Русский":    "Отвечай на русском языке.",
    }.get(language, "Respond in English.")

    try:
        result = run_async(
            pipeline.ask(message, response_instruction=lang_hint)
        )
    except RuntimeError as e:
        # Propagated from Generator (auth error, rate limit, etc.)
        return f"⚠️ **Service error:** {e}"
    except Exception as e:
        logger.exception("Unexpected error during pipeline.ask()")
        return f"⚠️ **Unexpected error:** {e}"

    if not result["answered"]:
        return "❌ This information is not available in the uploaded documents."

    sources = ""
    if result.get("sources"):
        sources = "\n\n**Sources:**\n" + "\n".join(
            f"- 📄 {s['source']} — Page {s['page']} (score: {s['score']:.2f})"
            for s in result["sources"]
        )

    score = result["score"]
    conf  = "🟢 High" if score >= 0.55 else "🟡 Medium" if score >= 0.30 else "🔴 Low"

    return (
        f"{result['answer']}{sources}\n\n"
        f"`Confidence: {conf} ({score:.0%}) · ⏱ {result['time']:.1f}s`"
    )


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background-color: #F0FDF4 !important; }
"""

with gr.Blocks(title="MediRAG", css=CSS, theme=gr.themes.Soft(primary_hue="green", neutral_hue="stone")) as demo:

    gr.Markdown("""
    # 🩺 MediRAG
    **Medical Document Intelligence System** — Ask questions about your medical documents.
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
    demo.launch()
