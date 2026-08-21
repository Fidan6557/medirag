"""
loader.py — Document Ingestion Module

Supported file types: PDF, DOCX, TXT, MD
For each document, returns text + metadata (file name, page, format).
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_PAGE_MARKER_RE = re.compile(r"^page\s+\d+\s*$", re.IGNORECASE)
_LOW_VALUE_PAGE_TITLES = {"index"}


def _normalise_line(line: str) -> str:
    return " ".join(line.split())


def _strip_repeated_pdf_lines(pages: List[Dict]) -> List[Dict]:
    """
    Removes repeated PDF headers/footers before chunking.

    Many PDFs repeat the publication title and page marker on every page.
    Leaving that text in each chunk makes retrieval match the header instead
    of the page content.
    """
    if not pages:
        return pages

    page_lines = []
    line_counts = Counter()

    for page in pages:
        lines = [_normalise_line(line) for line in page["text"].splitlines()]
        page_lines.append(lines)
        line_counts.update({line for line in lines if line})

    repeated_threshold = max(3, int(len(pages) * 0.75 + 0.999))
    repeated_lines = {
        line for line, count in line_counts.items() if count >= repeated_threshold
    }

    cleaned_pages = []
    for page, lines in zip(pages, page_lines, strict=False):
        kept_lines = []
        for line in lines:
            if not line:
                continue
            if line in repeated_lines:
                continue
            if _PAGE_MARKER_RE.match(line):
                continue
            kept_lines.append(line)

        text = "\n".join(kept_lines).strip()
        first_line = text.splitlines()[0].strip().lower() if text else ""
        if first_line in _LOW_VALUE_PAGE_TITLES:
            continue

        if text:
            cleaned_pages.append(
                {
                    "text": text,
                    "metadata": page["metadata"],
                }
            )

    return cleaned_pages


def load_pdf(file_path: str) -> List[Dict]:
    """
    Loads a PDF file and returns each page as a separate dictionary.
    Skips empty pages. Raises ValueError for encrypted/unreadable PDFs.
    """
    import fitz  # PyMuPDF

    pages = []
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF '{Path(file_path).name}': {e}") from e

    if doc.is_encrypted:
        doc.close()
        raise ValueError(
            f"'{Path(file_path).name}' is encrypted. Please provide an unprotected PDF."
        )

    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text().strip()
        except Exception as e:
            logger.warning(f"  Page {page_num} could not be read: {e}")
            continue

        if not text:
            continue  # skip blank pages

        pages.append(
            {
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": page_num,
                    "format": "pdf",
                },
            }
        )

    doc.close()
    pages = _strip_repeated_pdf_lines(pages)

    if not pages:
        raise ValueError(
            f"'{Path(file_path).name}' contains no extractable text. "
            "It may be a scanned image-only PDF."
        )

    return pages


def load_docx(file_path: str) -> List[Dict]:
    """
    Loads a DOCX file and returns its content as a single dictionary.
    """
    try:
        from docx import Document  # python-docx — lazy import

        doc = Document(file_path)
    except Exception as e:
        raise ValueError(f"Could not open DOCX '{Path(file_path).name}': {e}") from e

    full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    if not full_text.strip():
        raise ValueError(f"'{Path(file_path).name}' appears to be empty.")

    return [
        {
            "text": full_text,
            "metadata": {
                "source": Path(file_path).name,
                "page": 1,
                "format": "docx",
            },
        }
    ]


def load_txt(file_path: str) -> List[Dict]:
    """
    Loads TXT and MD files.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read().strip()
    except Exception as e:
        raise ValueError(f"Could not read '{Path(file_path).name}': {e}") from e

    if not text:
        raise ValueError(f"'{Path(file_path).name}' is empty.")

    fmt = Path(file_path).suffix.lstrip(".")  # "txt" or "md"

    return [
        {
            "text": text,
            "metadata": {
                "source": Path(file_path).name,
                "page": 1,
                "format": fmt,
            },
        }
    ]


def load_document(file_path: str) -> List[Dict]:
    """
    Automatically determines the file format and calls the right loader.
    This is the main entry point for document ingestion in the pipeline.
    """
    ext = Path(file_path).suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".md": load_txt,
    }

    if ext not in loaders:
        raise ValueError(
            f"Unsupported format: '{ext}'. Supported: {', '.join(loaders)}"
        )

    logger.info(f"Loading: {Path(file_path).name}")
    pages = loaders[ext](file_path)
    logger.info(f"  {len(pages)} page(s)/block(s) loaded from {Path(file_path).name}")

    return pages


def load_directory(dir_path: str) -> List[Dict]:
    """
    Loads all supported documents in a directory.
    Logs individual failures without stopping the whole batch.
    """
    supported = {".pdf", ".docx", ".txt", ".md"}
    all_pages = []
    dir_path_ = Path(dir_path)

    if not dir_path_.exists():
        logger.warning(f"Directory not found: {dir_path}")
        return []

    files = [f for f in dir_path_.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.warning(f"No supported documents found in: {dir_path}")
        return []

    logger.info(f"{len(files)} document(s) found — loading…")

    for file in files:
        try:
            pages = load_document(str(file))
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"  ✗ {file.name} — {e}")

    logger.info(f"Total {len(all_pages)} page(s)/block(s) loaded.\n")
    return all_pages
