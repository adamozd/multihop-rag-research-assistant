from pathlib import Path


def parse_pdf(path: Path) -> list[tuple[int, str]]:
    """Return one-based PDF page numbers and text in reading order."""
    import fitz

    with fitz.open(path) as document:
        pages = [(i + 1, page.get_text("text", sort=True)) for i, page in enumerate(document)]
    if not any(text.strip() for _, text in pages):
        raise ValueError(f"No extractable text in {path.name}; OCR is not implemented.")
    return pages
