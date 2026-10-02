import re
from ingestion.chunk import chunk_paper
from ingestion.chunk import sections
from ingestion.parse_pdf import parse_pdf


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


def test_sections_tokens_provenance_and_determinism():
    paper = {"paper_id": "p", "title": "Test paper", "url": "https://example.org/p"}
    pages = [(1, "Abstract\nalpha beta gamma\n\n1 Introduction\none two three four five six seven"),
             (2, "eight nine\n\n2 Methods\nten eleven twelve\nReferences\nExcluded bibliography")]
    chunks = chunk_paper(paper, pages, Tokenizer(), max_tokens=5)
    assert all(len(c.text.split()) <= 5 for c in chunks)
    assert {c.section for c in chunks} == {"Abstract", "1 Introduction", "2 Methods"}
    assert all("Excluded" not in c.text for c in chunks)
    assert any(c.page_end == 2 for c in chunks)
    assert chunks == chunk_paper(paper, pages, Tokenizer(), max_tokens=5)
    assert " ".join(c.text.replace("\n", " ") for c in chunks).split() == (
        "alpha beta gamma one two three four five six seven eight nine ten eleven twelve".split())


def test_real_pdf_extraction(tmp_path):
    import fitz
    path = tmp_path / "paper.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Abstract\nWe retrieve evidence.")
        document.save(path)
    pages = parse_pdf(path)
    assert pages[0][0] == 1 and "We retrieve evidence." in pages[0][1]


def test_numeric_table_rows_are_not_section_headings():
    result = sections([(1, "2 Results\n20      14.5\nResults show improvement.")])
    assert len(result) == 1
    assert "20" in result[0][1][0][1]
