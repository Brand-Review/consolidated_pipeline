"""Document loaders — dispatch by filename extension or mime type."""

from __future__ import annotations
import os
from typing import Union

from .base import RawDocument, Section, Page
from .pdf_loader import load_pdf
from .markdown_loader import load_markdown
from .html_loader import load_html
from .text_loader import load_text


_EXT_DISPATCH = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".txt": load_text,
}


def load_document(source: Union[str, bytes], filename: str) -> RawDocument:
    """Dispatch to the right loader based on filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    loader = _EXT_DISPATCH.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file extension: {ext} (filename={filename})")
    return loader(source, filename)


__all__ = [
    "RawDocument",
    "Section",
    "Page",
    "load_document",
    "load_pdf",
    "load_markdown",
    "load_html",
    "load_text",
]
