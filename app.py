"""
app.py - MediRAG Gradio UI
"""

import asyncio
import gradio as gr
from pathlib import Path
import tempfile
import os

from src.pipeline import MediRAGPipeline

pipeline = MediRAGPipeline()


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


def process_documents(files):
    if not files:
        return "⚠️ No files uploaded."

    pipeline.clear_knowledge_base()

    names = []
    for file in files:
        run_async(pipeline.ingest_document(file.name))
        names.append(Path(file.name).name)

    stats = pipeline.get_stats()
    return f"✅ {len(names)} document(s) processed — {stats['total_chunks']} chunks indexed."


def chat(message, history, language):
    lang_hint = {
        "English"    : "Respond in English.",
        "Azərbaycan" : "Cavabı Azərbaycan dilində ver.",
        "Русский"    : "Отвечай на русском языке."
    }.get(language, "Respond in English.")

    full_query = f"{message}\n\n[{lang_hint}]"
    result = run_async(pipeline.ask(full_query))

    if not result["answered"]:
        return "❌ This information is not available in the uploaded documents."

    sources = ""
    if result.get("sources"):
        sources = "\n\n**Sources:**\n" + "\n".join(
            f"- 📄 {s['source']} — Page {s['page']} (score: {s['score']:.2f})"
            for s in result["sources"]
        )

    score = result["score"]
    conf = "🟢 High" if score >= 0.70 else "🟡 Medium" if score >= 0.50 else "🔴 Low"

    return f"{result['answer']}{sources}\n\n`Confidence: {conf} ({score:.0%}) · ⏱ {result['time']:.1f}s`"


with gr.Blocks(
    title="MediRAG",
    css="""
    body { background-color: #F0FDF4 !important; }
    .gradio-container { background-color: #F0FDF4 !important; }
    """
) as demo:

    gr.Markdown("""
    # 🩺 MediRAG
    **Medical Document Intelligence System** — Ask questions about your medical documents.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Documents")
            files = gr.File(
                label="Upload Documents",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                file_count="multiple"
            )
            language = gr.Dropdown(
                choices=["English", "Azərbaycan", "Русский"],
                value="English",
                label="🌍 Language"
            )
            process_btn = gr.Button("⚙️ Process Documents", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

            process_btn.click(
                fn=process_documents,
                inputs=[files],
                outputs=[status]
            )

        with gr.Column(scale=3):
            gr.Markdown("### 💬 Chat")
            gr.ChatInterface(
                fn=chat,
                additional_inputs=[language],
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    scale=7
                ),
            )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(
        primary_hue="green",
        neutral_hue="stone",
    ))