"""Block-based reading order for ordinary one- and two-column academic PDFs."""
from collections import Counter
from pathlib import Path
import re

from ingestion.headings import is_heading


class PageText(str):
    """Text with layout-verified section labels; ordinary string consumers still work."""
    def __new__(cls, text, headings):
        value = super().__new__(cls, text)
        value.headings = frozenset(headings)
        return value


def _heading(block: dict) -> str | None:
    lines = ["".join(s["text"] for s in line["spans"]).strip() for line in block["lines"]]
    candidate = " ".join(lines)
    if len(lines) > 3 or not is_heading(candidate):
        return None
    spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
    bold = all(any(weight in s["font"].lower() for weight in ("bold", "medi")) for s in spans)
    title = re.sub(r"^(?:[\d.]+|[IVX]+\.|[A-Z](?:\.\d+)*\.?)\s+", "", candidate)
    if bold or title.isupper() or (len(lines) == 1 and candidate[:1].isupper()
                                  and not re.match(r"(?:\d|[IVX]+\.|[A-Z](?:\.\d+)*\.?\s)", candidate)):
        return candidate
    return None


def _text(block: dict) -> str:
    lines = ["".join(s["text"] for s in line["spans"]).strip() for line in block["lines"]]
    # PDF producers often store a section number and wrapped title on separate lines.
    heading = _heading(block)
    if heading:
        return heading
    return "\n".join(lines)


def _reading_order(blocks: list[dict], width: float) -> list[dict]:
    # Infer columns only from substantial paragraphs, not equation/table cells.
    body = [b for b in blocks if len(_text(b)) >= 100 and b["bbox"][2] - b["bbox"][0] > width * .22]
    middle = width / 2
    left = [b for b in body if b["bbox"][2] <= middle + 5]
    right = [b for b in body if b["bbox"][0] >= middle - 5]
    if not (len(left) >= 2 and len(right) >= 2
            and sum(len(_text(b)) for b in left + right) > sum(len(_text(b)) for b in body) * .6):
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    gutter_left = max(b["bbox"][2] for b in left)
    gutter_right = min(b["bbox"][0] for b in right)
    if gutter_right <= gutter_left:
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    split = (gutter_left + gutter_right) / 2
    # Full-width titles, figures and captions divide a page into reading bands.
    spanning = sorted([b for b in blocks if b["bbox"][0] < gutter_left - 10
                       and b["bbox"][2] > gutter_right + 10], key=lambda b: b["bbox"][1])
    remaining = [b for b in blocks if b not in spanning]
    output = []
    for barrier in spanning + [None]:
        band = [b for b in remaining if barrier is None or b["bbox"][1] < barrier["bbox"][1]]
        output.extend(sorted(band, key=lambda b: ((b["bbox"][0] + b["bbox"][2]) / 2 >= split,
                                                  b["bbox"][1], b["bbox"][0])))
        remaining = [b for b in remaining if b not in band]
        if barrier is not None:
            output.append(barrier)
    return output


def parse_pdf(path: Path) -> list[tuple[int, str]]:
    """Preserve paragraph boundaries and one-based PDF page provenance; no OCR."""
    import pymupdf

    with pymupdf.open(path) as document:
        extracted = [(page.rect.width, page.rect.height,
                      [b for b in page.get_text("dict", sort=False)["blocks"] if b["type"] == 0])
                     for page in document]
    def margin(block, height):
        return block["bbox"][3] < height * .07 or block["bbox"][1] > height * .93
    repeated = Counter(text for _, height, blocks in extracted
                       for text in {_text(b) for b in blocks if margin(b, height)})
    pages = []
    for i, (width, height, blocks) in enumerate(extracted):
        blocks = [b for b in blocks if not (margin(b, height)
                  and (re.fullmatch(r"\d+", _text(b)) or repeated[_text(b)] >= max(2, len(extracted) // 2)))]
        pages.append((i + 1, PageText("\n\n".join(_text(b) for b in _reading_order(blocks, width)),
                                      [h for b in blocks if (h := _heading(b))])))
    if not any(text.strip() for _, text in pages):
        raise ValueError(f"No extractable text in {path.name}; OCR is not implemented.")
    return pages
