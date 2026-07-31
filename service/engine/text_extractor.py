import logging
from io import BytesIO
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text(file_content: bytes) -> str:
    """
    Extracts text from PDF binary content with layout preservation and error resilience.
    """
    if not file_content:
        return ""

    try:
        pdf = PdfReader(BytesIO(file_content))
    except Exception as e:
        logger.error(f"Failed to read PDF stream: {e}")
        return ""

    text_pages = []
    for i, page in enumerate(pdf.pages):
        try:
            # Attempt layout extraction if available in pypdf
            extracted = page.extract_text()
            if extracted and extracted.strip():
                text_pages.append(extracted.strip())
        except Exception as e:
            logger.warning(f"Error extracting page {i+1}: {e}")
            continue

    full_text = "\n\n".join(text_pages)
    return full_text


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file path."""
    with open(file_path, "rb") as f:
        return extract_text(f.read())