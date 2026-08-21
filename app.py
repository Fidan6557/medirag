"""MediRAG Gradio application."""

import asyncio
import logging
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

import gradio as gr

from config import (
    GRADIO_SHARE,
    MAX_FILE_SIZE_MB,
    SERVER_NAME,
    SERVER_PORT,
)
from src.presentation import (
    escape_markdown,
    format_index_status,
    format_processing_status,
)

if TYPE_CHECKING:
    from src.pipeline import MediRAGPipeline

logger = logging.getLogger(__name__)
_pipeline: Optional["MediRAGPipeline"] = None
_pipeline_lock = Lock()


def configure_logging() -> None:
    """Configure concise application logging without affecting library imports."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy_logger in (
        "httpx",
        "httpcore",
        "sentence_transformers",
        "huggingface_hub",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_pipeline() -> "MediRAGPipeline":
    """Initialise heavy ML dependencies only when they are first needed."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from src.pipeline import MediRAGPipeline

                _pipeline = MediRAGPipeline()
    return _pipeline


def _file_path(upload: Any) -> Path:
    """Normalise Gradio upload values to a filesystem path."""
    return Path(getattr(upload, "name", upload))


# ── Callbacks ────────────────────────────────────────────────────────────────


async def process_documents(files: Optional[list[Any]]) -> str:
    """Validate uploads and replace the current knowledge base safely."""
    if not files:
        return "### ⚠️ No files selected\nUpload at least one supported document first."

    valid_paths: list[str] = []
    preflight_results: list[dict] = []
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    for upload in files:
        path = _file_path(upload)
        try:
            size = path.stat().st_size
        except OSError as exc:
            preflight_results.append(
                {
                    "success": False,
                    "file": path.name,
                    "error": f"file is unavailable ({exc})",
                }
            )
            continue

        if size > max_bytes:
            preflight_results.append(
                {
                    "success": False,
                    "file": path.name,
                    "error": f"larger than the {MAX_FILE_SIZE_MB} MB limit",
                }
            )
            continue
        valid_paths.append(str(path))

    if not valid_paths:
        return format_processing_status(preflight_results, total_chunks=0)

    try:
        pipeline = get_pipeline()
        results = await pipeline.replace_documents(valid_paths)
        results.extend(preflight_results)
        total_chunks = pipeline.vectorstore.count()
        return format_processing_status(results, total_chunks)
    except Exception:
        logger.exception("Document batch processing failed")
        return (
            "### ⚠️ Processing failed\n"
            "The documents could not be indexed. Check the application logs for details."
        )


async def knowledge_base_status() -> str:
    """Report whether a persisted index is available without loading the model."""
    try:
        pipeline = get_pipeline()
        total_chunks = await asyncio.to_thread(pipeline.vectorstore.count)
        return format_index_status(total_chunks)
    except Exception:
        logger.exception("Could not inspect the knowledge base")
        return "### ⚠️ Index unavailable\nCheck the application logs for details."


async def clear_documents() -> tuple[None, str]:
    """Clear the local index and reset the upload control."""
    try:
        pipeline = get_pipeline()
        await asyncio.to_thread(pipeline.clear_knowledge_base)
    except Exception:
        logger.exception("Could not clear the knowledge base")
        return None, "### ⚠️ Clear failed\nCheck the application logs for details."
    return None, "### Knowledge base is empty\nUpload documents to begin."


async def chat(message: str, _history: list, language: str) -> str:
    """Answer one question using the currently indexed documents."""
    lang_hint = {
        "English": "Respond in English.",
        "Azərbaycan": "Cavabı Azərbaycan dilində ver.",
        "Русский": "Отвечай на русском языке.",
    }.get(language, "Respond in English.")

    try:
        pipeline = get_pipeline()
        result = await pipeline.ask(message, response_instruction=lang_hint)
    except ValueError as exc:
        return f"⚠️ {escape_markdown(exc)}"
    except RuntimeError as exc:
        return f"⚠️ **Service unavailable:** {escape_markdown(exc)}"
    except Exception:
        logger.exception("Unexpected error during question answering")
        return "⚠️ **Something went wrong.** Check the application logs and try again."

    if not result["answered"]:
        return f"ℹ️ {result['answer']}"

    source_lines = []
    for source in result.get("sources", []):
        source_name = escape_markdown(Path(str(source["source"])).name)
        page = source.get("page", "—")
        score = max(0.0, min(1.0, float(source.get("score", 0.0))))
        source_lines.append(
            f"- 📄 `{source_name}` · page {page} · {score:.0%} retrieval match"
        )

    evidence = ""
    if source_lines:
        evidence = "\n\n---\n**Supporting excerpts**\n" + "\n".join(source_lines)

    score = max(0.0, min(1.0, float(result["score"])))
    match_label = "Strong" if score >= 0.55 else "Moderate"
    metadata = (
        f"\n\n`{match_label} retrieval match · {score:.0%} · " f"{result['time']:.1f}s`"
    )
    return f"{result['answer']}{evidence}{metadata}"


# ── Interface ────────────────────────────────────────────────────────────────

CSS = """
:root {
    --med-bg: #f4f8f7;
    --med-surface: rgba(255, 255, 255, 0.94);
    --med-ink: #102a2a;
    --med-muted: #5d7473;
    --med-border: #d7e5e2;
    --med-brand: #087f6d;
    --med-brand-dark: #075f54;
}

body,
.gradio-container {
    background:
        radial-gradient(circle at 8% 4%, rgba(13, 148, 136, 0.10), transparent 32rem),
        radial-gradient(circle at 94% 12%, rgba(14, 116, 144, 0.08), transparent 28rem),
        var(--med-bg) !important;
    color: var(--med-ink);
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 28px 22px 36px !important;
}

#medirag-hero {
    padding: 26px 30px;
    margin-bottom: 18px;
    border: 1px solid rgba(8, 127, 109, 0.16);
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(255,255,255,.98), rgba(239,250,247,.94));
    box-shadow: 0 18px 48px rgba(26, 73, 68, 0.09);
}

#medirag-hero .eyebrow {
    margin: 0 0 8px;
    color: var(--med-brand);
    font-size: 0.74rem;
    font-weight: 760;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

#medirag-hero h1 {
    margin: 0;
    color: var(--med-ink);
    font-size: clamp(2rem, 5vw, 3.25rem);
    line-height: 1.02;
    letter-spacing: -0.045em;
}

#medirag-hero .lede {
    max-width: 720px;
    margin: 12px 0 0;
    color: var(--med-muted);
    font-size: 1.02rem;
    line-height: 1.6;
}

#medirag-hero .capabilities {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

#medirag-hero .capabilities span {
    padding: 6px 10px;
    border: 1px solid var(--med-border);
    border-radius: 999px;
    background: rgba(255,255,255,.76);
    color: #315957;
    font-size: 0.78rem;
    font-weight: 650;
}

.workspace-panel {
    border: 1px solid var(--med-border) !important;
    border-radius: 18px !important;
    background: var(--med-surface) !important;
    box-shadow: 0 12px 34px rgba(29, 68, 65, 0.07) !important;
}

#control-panel {
    padding: 18px !important;
}

#chat-panel {
    padding: 12px 14px 14px !important;
}

#upload-zone {
    border: 1px dashed #91bdb6 !important;
    border-radius: 14px !important;
    background: #f8fcfb !important;
}

#knowledge-status {
    min-height: 118px;
    padding: 12px 14px;
    border: 1px solid #dceae7;
    border-radius: 12px;
    background: #f7fbfa;
}

#medical-note {
    margin-top: 16px;
    padding: 13px 16px;
    border-left: 3px solid #d79a27;
    border-radius: 8px;
    background: #fffaf0;
    color: #66532e;
    font-size: 0.86rem;
}

#privacy-note {
    color: var(--med-muted);
    font-size: 0.78rem;
    line-height: 1.5;
}

button.primary {
    border-color: var(--med-brand) !important;
    background: var(--med-brand) !important;
}

button.primary:hover {
    border-color: var(--med-brand-dark) !important;
    background: var(--med-brand-dark) !important;
}

@media (max-width: 760px) {
    .gradio-container {
        padding: 14px 10px 24px !important;
    }

    #medirag-hero {
        padding: 21px 19px;
        border-radius: 17px;
    }

    #medirag-hero .lede {
        font-size: 0.94rem;
    }
}
"""

HERO = """
<section id="medirag-hero">
  <p class="eyebrow">Document-grounded medical intelligence</p>
  <h1>MediRAG</h1>
  <p class="lede">
    Search your medical documents and receive concise answers grounded in the
    passages most relevant to your question.
  </p>
  <div class="capabilities">
    <span>PDF · DOCX · TXT · MD</span>
    <span>English · Azərbaycan · Русский</span>
    <span>Source & page references</span>
    <span>Local vector index</span>
  </div>
</section>
"""

with gr.Blocks(
    title="MediRAG · Medical Document Assistant",
    analytics_enabled=False,
) as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=290, elem_classes="workspace-panel"):
            with gr.Group(elem_id="control-panel"):
                gr.Markdown(
                    "## Knowledge base\n"
                    "Add one or more documents, then build a searchable index."
                )
                files_input = gr.File(
                    label="Documents",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    file_count="multiple",
                    type="filepath",
                    height=150,
                    elem_id="upload-zone",
                )
                language = gr.Dropdown(
                    choices=["English", "Azərbaycan", "Русский"],
                    value="English",
                    label="Answer language",
                )
                with gr.Row():
                    process_btn = gr.Button(
                        "Build knowledge base",
                        variant="primary",
                        scale=3,
                    )
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                status = gr.Markdown(
                    "### Knowledge base is empty\nUpload documents to begin.",
                    elem_id="knowledge-status",
                )
                gr.Markdown(
                    f"Files are limited to **{MAX_FILE_SIZE_MB} MB each**. Document "
                    "text stays in the local ChromaDB index; selected excerpts and "
                    "questions are sent to Groq.",
                    elem_id="privacy-note",
                )

        with gr.Column(scale=8, min_width=420, elem_classes="workspace-panel"):
            with gr.Group(elem_id="chat-panel"):
                gr.Markdown("## Ask your documents")
                gr.ChatInterface(
                    fn=chat,
                    additional_inputs=[language],
                    chatbot=gr.Chatbot(
                        height=510,
                        layout="bubble",
                        buttons=["copy", "copy_all"],
                        placeholder=(
                            "Build a knowledge base, then ask a focused question."
                        ),
                    ),
                    textbox=gr.Textbox(
                        placeholder="Ask a question about the indexed documents…",
                        scale=7,
                        container=False,
                    ),
                    examples=[
                        [
                            "Summarise the main clinical recommendations.",
                            "English",
                        ],
                        [
                            "What dosage guidance does the document provide?",
                            "English",
                        ],
                        [
                            "Which warnings or contraindications are mentioned?",
                            "English",
                        ],
                    ],
                    example_labels=[
                        "Key recommendations",
                        "Dosage guidance",
                        "Safety warnings",
                    ],
                    flagging_mode="never",
                    api_visibility="private",
                    fill_height=True,
                )

    gr.Markdown(
        "**Medical safety notice:** MediRAG can be incomplete or wrong. Verify "
        "important information in the original source and do not use this tool as "
        "a substitute for diagnosis, treatment, or professional medical advice.",
        elem_id="medical-note",
    )

    process_btn.click(
        fn=process_documents,
        inputs=[files_input],
        outputs=[status],
        api_name=False,
    )
    clear_btn.click(
        fn=clear_documents,
        outputs=[files_input, status],
        api_name=False,
    )
    demo.load(
        fn=knowledge_base_status,
        outputs=[status],
        api_name=False,
    )


if __name__ == "__main__":
    configure_logging()
    demo.queue(default_concurrency_limit=1, max_size=20).launch(
        css=CSS,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=GRADIO_SHARE,
        max_file_size=f"{MAX_FILE_SIZE_MB}mb",
        show_error=False,
        footer_links=[],
    )
