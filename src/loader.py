"""
loader.py — Document Ingestion Module

Supported file types: PDF, DOCX, TXT, MD
For each document, it returns text + metadata (file name, page, format).
"""

import os
from pathlib import Path
from typing import List, Dict
import fitz                        # PyMuPDF — PDF reading


def load_pdf(file_path: str) -> List[Dict]:
    """
    Loads a PDF file and returns each page as a separate dictionary.

    Returns:
        [
            {
                "text": "page text...",
                "metadata": {
                    "source": "file_name.pdf",
                    "page": 1,
                    "format": "pdf"
                }
            },
            ...
        ]
    """
    pages = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if not text:          # Skip empty pages
            continue

        pages.append({
            "text": text,
            "metadata": {
                "source": Path(file_path).name,
                "page": page_num,
                "format": "pdf"
            }
        })

    doc.close()
    return pages


def load_docx(file_path: str) -> List[Dict]:
    """
    Loads a DOCX file and returns its content as a single dictionary.
    """
    from docx import Document      # python-docx — imported lazily to avoid a DLL
                                    # conflict with torch when loaded before it

    doc = Document(file_path)
    full_text = "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    )

    return [{
        "text": full_text,
        "metadata": {
            "source": Path(file_path).name,
            "page": 1,
            "format": "docx"
        }
    }]


def load_txt(file_path: str) -> List[Dict]:
    """
    Loads TXT and MD files.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    fmt = Path(file_path).suffix.lstrip(".")   # "txt" or "md"

    return [{
        "text": text,
        "metadata": {
            "source": Path(file_path).name,
            "page": 1,
            "format": fmt
        }
    }]


def load_document(file_path: str) -> List[Dict]:
    """
    Automatically determines the file's format and calls the appropriate loader.
    This function is the main entry point for document ingestion in the pipeline.
    """
    ext = Path(file_path).suffix.lower()

    loaders = {
        ".pdf":  load_pdf,
        ".docx": load_docx,
        ".txt":  load_txt,
        ".md":   load_txt,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported format: {ext}")

    print(f"Loading: {Path(file_path).name}")
    pages = loaders[ext](file_path)
    print(f"  {len(pages)} pages/blocks found")

    return pages


def load_directory(dir_path: str) -> List[Dict]:
    """
    Loads all supported documents in the directory.
    """
    supported = {".pdf", ".docx", ".txt", ".md"}
    all_pages = []

    files = [
        f for f in Path(dir_path).iterdir()
        if f.suffix.lower() in supported
    ]

    if not files:
        print(f"{dir_path} directory contains no supported documents.")
        return []

    print(f"\n{len(files)} documents found — loading...\n")

    for file in files:
        try:
            pages = load_document(str(file))
            all_pages.extend(pages)
        except Exception as e:
            print(f"   {file.name} failed to load  : {e}")

    print(f"\nTotal {len(all_pages)} pages/blocks loaded.\n")
    return all_pages            