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


def test_two_columns_and_spanning_title(tmp_path):
    import pymupdf
    path = tmp_path / "columns.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=800)
        # Deliberately draw right column first and interleave content-object order.
        for x, y, label in [(320, 110, "RIGHT FIRST"), (50, 310, "LEFT SECOND"),
                            (320, 310, "RIGHT SECOND"), (50, 110, "LEFT FIRST")]:
            page.insert_textbox(pymupdf.Rect(x, y, x + 220, y + 160),
                                label + "\n" + "Evidence paragraph has many meaningful words. " * 5, fontsize=10)
        page.insert_text((50, 60), "A full width paper title about retrieval and reasoning across documents", fontsize=14)
        page.insert_text((50, 510), "3.1\nRetrieval Method", fontname="hebo", fontsize=12)
        doc.save(path)
    text = parse_pdf(path)[0][1]
    assert text.index("LEFT FIRST") < text.index("LEFT SECOND") < text.index("RIGHT FIRST") < text.index("RIGHT SECOND")
    assert text.index("A full width") < text.index("LEFT FIRST")
    assert "3.1 Retrieval Method" in text.headings


def test_single_column_tables_and_repeated_margins(tmp_path):
    import pymupdf
    path = tmp_path / "single.pdf"
    with pymupdf.open() as doc:
        for i in range(2):
            page = doc.new_page(width=600, height=800)
            page.insert_text((60, 30), "Preprint header")
            page.insert_text((60, 100), "3.3\nSELF-RAG INFERENCE", fontsize=12)
            page.insert_textbox(pymupdf.Rect(60, 150, 540, 260),
                                "Adaptive retrieval with a threshold is optional. " * 6, fontsize=10)
            page.insert_text((60, 300), "0.25 Retrieve top1 41.8 73.1 28.6\n11 of 50 states Relevant")
            page.insert_text((300, 780), str(i + 1))
        doc.save(path)
    pages = parse_pdf(path)
    assert all("Preprint header" not in text for _, text in pages)
    assert all("0.25 Retrieve" in text for _, text in pages)
    labels = [label for label, _ in sections(pages)]
    assert labels == ["3.3 SELF-RAG INFERENCE", "3.3 SELF-RAG INFERENCE"]


def test_appendix_after_references_is_retained(tmp_path):
    import pymupdf
    path = tmp_path / "appendix.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((60, 100), "References", fontname="hebo")
        page.insert_text((60, 140), "A bibliography title without a final period")
        page.insert_text((60, 200), "A\nAdditional Methods", fontname="hebo")
        page.insert_text((60, 240), "The appendix contains evidence about retrieval.")
        doc.save(path)
    chunks = chunk_paper({"paper_id": "p", "title": "Paper", "url": "https://example.org"},
                         parse_pdf(path), Tokenizer())
    assert len(chunks) == 1 and chunks[0].section == "A Additional Methods"
    assert "bibliography" not in chunks[0].text
