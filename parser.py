# parser.py
"""
PDF text extraction helpers.
- Primary engine: PyMuPDF (fitz) — fast & reliable layout-aware text.
- Fallback: pdfminer.six — used automatically if PyMuPDF fails/returns empty.
Also includes simple email/name extraction and text hashing.

Usage:
    from parser import extract_text_from_pdf, extract_text_from_pdf_bytes, extract_email_and_name, clean_text
"""

from __future__ import annotations
import re
import hashlib
from typing import Tuple, Optional

# --- PyMuPDF (primary) ---
try:
    import fitz  # PyMuPDF
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

# --- pdfminer.six (fallback) ---
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    _HAVE_PDFMINER = True
except Exception:
    _HAVE_PDFMINER = False


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control chars for more stable downstream processing."""
    if not text:
        return ""
    # Collapse weird whitespace, keep newlines (useful for detecting names at top).
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_with_pymupdf_path(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)


def _extract_with_pymupdf_bytes(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)


def _extract_with_pdfminer_path(path: str) -> str:
    return pdfminer_extract_text(path)


def _extract_with_pdfminer_bytes(data: bytes) -> str:
    # pdfminer accepts file-like objects; we'll use BytesIO
    from io import BytesIO
    bio = BytesIO(data)
    return pdfminer_extract_text(bio)


def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF at 'path' using PyMuPDF with pdfminer fallback.
    Returns cleaned text (string, may be empty if PDF truly has no text layer).
    """
    text = ""
    if _HAVE_PYMUPDF:
        try:
            text = _extract_with_pymupdf_path(path)
        except Exception:
            text = ""
    if not text and _HAVE_PDFMINER:
        try:
            text = _extract_with_pdfminer_path(path)
        except Exception:
            text = ""
    return clean_text(text)


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Extract text from PDF bytes (useful for Streamlit uploads).
    Returns cleaned text.
    """
    text = ""
    if _HAVE_PYMUPDF:
        try:
            text = _extract_with_pymupdf_bytes(data)
        except Exception:
            text = ""
    if not text and _HAVE_PDFMINER:
        try:
            text = _extract_with_pdfminer_bytes(data)
        except Exception:
            text = ""
    return clean_text(text)


_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")

def extract_email_and_name(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Very simple email + name heuristic:
      - Email: first email pattern in text.
      - Name: first non-empty line near the top that looks like 2–4 capitalized words.
    This is intentionally simple; you can swap in smarter NER later.
    """
    email = None
    m = _EMAIL_RE.search(text)
    if m:
        email = m.group(1)

    # Heuristic name detection (top ~10 lines)
    name = None
    top = text.splitlines()[:12]
    # Common headers to skip
    bad_starts = {"resume", "curriculum vitae", "cv", "contact", "profile", "summary"}
    for line in top:
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(low.startswith(b) for b in bad_starts):
            continue
        # Look for 2–4 capitalized tokens (e.g., "Jane Doe", "A. B. Smith")
        tokens = [t for t in re.split(r"[\s,;]+", s) if t]
        caps = sum(bool(re.match(r"^[A-Z][A-Za-z\.\-']*$", t)) for t in tokens)
        if 2 <= caps <= 4:
            name = s
            break

    return email, name


def hash_text(text: str) -> str:
    """MD5 hash of normalized text to help dedupe."""
    return hashlib.md5(clean_text(text).encode("utf-8")).hexdigest()
