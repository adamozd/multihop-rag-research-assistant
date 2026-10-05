"""Conservative section labels shared by extraction and chunking."""
import re

HEADER = re.compile(
    r"^(?:(?:[1-9]\d?(?:\.\d+)*\.?|[IVX]+\.|[A-Z](?:\.\d+)*\.?)\s+[A-Za-z].{1,100}|"
    r"(?:abstract|introduction|background|related work|method(?:s|ology)?|approach|"
    r"experiments?(?: and results)?|results?(?: and discussion)?|discussion|conclusions?|"
    r"limitations|references|acknowledg(?:e)?ments|appendix)\s*[:.]?)$", re.I
)


def is_heading(text: str) -> bool:
    # Decimal table cells and long sentence fragments are not section labels.
    return bool(HEADER.fullmatch(text) and len(text.split()) <= 14
                and not re.search(r"\d\s+\d|\d+\.\d+.*\d+\.\d+", text)
                and not (text.endswith('.') and len(text.split()) > 4))
