"""
PDF Processor: Download, extract text, and chunk research papers.
"""
import io
import os
import re
import tempfile
import requests
from pathlib import Path

# Try to import pypdf, fallback to PyPDF2
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("pypdf or PyPDF2 is required. Install with: pip install pypdf")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def normalize_pdf_url(url: str) -> str:
    """
    Ensure the URL points directly to the PDF file rather than the abstract or HTML landing page.
    Handles arXiv abstract URLs, HTML URLs, and missing extensions.
    """
    if not url:
        return ""

    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    # Handle arXiv abstract or html URLs
    if "arxiv.org/abs/" in url:
        url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")
    elif "arxiv.org/html/" in url:
        url = url.replace("arxiv.org/html/", "arxiv.org/pdf/")

    # Ensure arXiv PDF URLs have .pdf extension
    if "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        url = f"{url}.pdf"

    return url


def download_pdf(url: str, timeout: int = 30) -> bytes:
    """
    Download a PDF from a URL.

    Args:
        url: The URL to download the PDF from
        timeout: Request timeout in seconds

    Returns:
        PDF content as bytes

    Raises:
        requests.RequestException: If download fails
        ValueError: If returned content is not a valid PDF
    """
    direct_url = normalize_pdf_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/octet-stream,*/*"
    }

    response = requests.get(direct_url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()

    content = response.content
    if not content:
        raise ValueError("Downloaded file is empty")

    # Check for HTML responses (landing pages or error pages)
    sample = content[:1024].lstrip()
    if sample.startswith(b"<!DOCTYPE") or sample.startswith(b"<html") or sample.startswith(b"<HTML"):
        raise ValueError("Server returned an HTML webpage instead of a PDF document.")

    # PDF files start with %PDF
    if b"%PDF" not in sample:
        # Some servers send leading bytes or BOM, check further in header
        if b"%PDF" not in content[:2048]:
            raise ValueError("Downloaded content does not appear to be a valid PDF.")

    return content


def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extract text from PDF content.

    Args:
        pdf_content: PDF file content as bytes

    Returns:
        Extracted text string

    Raises:
        ValueError: If PDF cannot be read
    """
    try:
        # Use io.BytesIO or temp file
        stream = io.BytesIO(pdf_content)
        reader = PdfReader(stream)
        text_parts = []

        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text.strip())
            except Exception as e:
                print(f"Warning: Failed to extract text from page {page_num + 1}: {e}")
                continue

        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            raise ValueError("No extractable text could be found in the PDF (may be a scanned image)")

        return full_text

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of chunk dictionaries with text and position info
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text.strip(),
                "start": start,
                "end": end,
                "chunk_id": len(chunks)
            })

        # Move start position, accounting for overlap
        start += chunk_size - overlap

        # Safety: ensure progress
        if start <= text_length and len(chunks) > 0 and start <= chunks[-1]["start"]:
            start = chunks[-1]["start"] + 1

    return chunks


def process_paper(url: str) -> list[dict]:
    """
    Complete pipeline: download PDF, extract text, and chunk.

    Args:
        url: URL of the paper PDF

    Returns:
        List of text chunks

    Raises:
        ValueError: If any step fails
    """
    try:
        pdf_content = download_pdf(url)
    except requests.RequestException as e:
        raise ValueError(f"Failed to download PDF: {str(e)}")

    # Check size (max 50MB)
    if len(pdf_content) > 50 * 1024 * 1024:
        raise ValueError("PDF file too large (max 50MB)")

    if len(pdf_content) < 100:
        raise ValueError("Downloaded file is too small to be a valid PDF")

    text = extract_text_from_pdf(pdf_content)

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Failed to create text chunks from PDF")

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_url = sys.argv[1]
        print(f"Processing: {test_url}")
        try:
            res_chunks = process_paper(test_url)
            print(f"Extracted {len(res_chunks)} chunks")
            for i, chunk in enumerate(res_chunks[:3]):
                print(f"\n--- Chunk {i} ---")
                print(chunk["text"][:200] + "...")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python pdf_processor.py <PDF_URL>")
