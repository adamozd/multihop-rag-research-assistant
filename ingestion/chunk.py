"""Section boundaries first, paragraph packing second, token splitting last."""
import hashlib
import re

from reasoning.models import Evidence

from ingestion.headings import is_heading


def sections(pages: list[tuple[int, str]]) -> list[tuple[str, list[tuple[int, str]]]]:
    output = []
    name, blocks = "Preamble", []
    for page, text in pages:
        paragraph = []
        for line in text.splitlines() + [""]:
            line = line.strip()
            heading = line in text.headings if hasattr(text, "headings") else is_heading(line)
            if not line or heading:
                if paragraph:
                    blocks.append((page, " ".join(paragraph)))
                    paragraph = []
                if heading:
                    if blocks:
                        output.append((name, blocks))
                    name, blocks = line, []
            else:
                paragraph.append(line)
    if blocks:
        output.append((name, blocks))
    return output


def chunk_paper(paper: dict, pages: list[tuple[int, str]], tokenizer, max_tokens: int = 480) -> list[Evidence]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    chunks = []
    for section, blocks in sections(pages):
        if re.match(r"^(?:\d+\.?\s+)?(?:references|acknowledg)", section, re.I):
            continue
        current, current_pages = [], []

        def flush():
            if not current:
                return
            text = "\n\n".join(current)
            identity = f"{paper['paper_id']}|{section}|{current_pages}|{len(chunks)}|{text}"
            chunk_id = "c_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            chunks.append(Evidence(chunk_id=chunk_id, paper_id=paper["paper_id"], title=paper["title"],
                                   url=paper["url"], section=section, page_start=min(current_pages),
                                   page_end=max(current_pages), text=text))
            current.clear()
            current_pages.clear()

        def count(text):
            return len(tokenizer.encode(text, add_special_tokens=False))

        for page, block in blocks:
            # Offset slices preserve original evidence wording, unlike decode/re-encode.
            encoded = tokenizer(block, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            pieces = []
            start = 0
            while start < len(offsets):
                end = min(start + max_tokens, len(offsets))
                piece = block[offsets[start][0]:offsets[end - 1][1]]
                # A wordpiece cut can re-tokenize into more tokens at a new boundary.
                while count(piece) > max_tokens and end > start + 1:
                    end -= 1
                    piece = block[offsets[start][0]:offsets[end - 1][1]]
                if count(piece) > max_tokens:
                    raise ValueError("Token budget too small to preserve this text span")
                pieces.append(piece)
                start = end
            for piece in pieces:
                if current and count("\n\n".join(current + [piece])) > max_tokens:
                    flush()
                current.append(piece)
                current_pages.append(page)
        flush()
    return chunks
