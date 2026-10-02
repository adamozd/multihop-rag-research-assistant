import argparse
import json
from pathlib import Path

from ingestion.chunk import chunk_paper
from ingestion.parse_pdf import parse_pdf
from retrieval.embed import embedding_model
from retrieval.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Fetch and index the curated arXiv corpus or a topic search.")
    parser.add_argument("--query", help="arXiv query; overrides curated corpus")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("data/papers"))
    parser.add_argument("--reindex-local", action="store_true", help="Rebuild chunks from the existing manifest and PDFs")
    args = parser.parse_args()
    if not 15 <= args.limit <= 30:
        parser.error("--limit must be between 15 and 30")
    if args.reindex_local:
        manifest_path = args.output / "manifest.json"
        saved = json.loads(manifest_path.read_text())
        retriever = Retriever()
        tokenizer = embedding_model().tokenizer
        for paper in saved["papers"]:
            chunks = chunk_paper(paper, parse_pdf(Path(paper["pdf"])), tokenizer)
            retriever.replace_paper(paper["paper_id"], chunks)
            paper["chunks"] = len(chunks)
            print(f"Reindexed {paper['paper_id']}: {len(chunks)} chunks", flush=True)
        manifest_path.write_text(json.dumps(saved, indent=2))
        return
    import arxiv
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(__file__).with_name("corpus.json").read_text())
    search = (arxiv.Search(query=args.query, max_results=args.limit, sort_by=arxiv.SortCriterion.Relevance)
              if args.query else arxiv.Search(id_list=[p["id"] for p in manifest["papers"]]))
    retriever = Retriever()
    tokenizer = embedding_model().tokenizer
    records, failures = [], []
    for result in arxiv.Client(page_size=30, delay_seconds=3, num_retries=3).results(search):
        version_id = result.get_short_id()
        paper_id = version_id.rsplit("v", 1)[0]
        filename = version_id.replace("/", "_") + ".pdf"
        path = args.output / filename
        try:
            if not path.exists():
                temporary = filename + ".part"
                result.download_pdf(dirpath=str(args.output), filename=temporary)
                (args.output / temporary).replace(path)
            paper = {"paper_id": paper_id, "version_id": version_id, "title": result.title,
                     "url": f"https://arxiv.org/abs/{version_id}", "authors": [a.name for a in result.authors]}
            chunks = chunk_paper(paper, parse_pdf(path), tokenizer)
            retriever.replace_paper(paper_id, chunks)
            records.append({**paper, "chunks": len(chunks), "pdf": str(path)})
            print(f"Indexed {paper_id}: {len(chunks)} chunks — {result.title}", flush=True)
        except Exception as exc:
            failures.append({"paper_id": paper_id, "error": str(exc)})
            print(f"Failed {paper_id}: {exc}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps({"papers": records, "failures": failures}, indent=2))
    if failures or len(records) < 15:
        raise SystemExit("Corpus incomplete; inspect data/papers/manifest.json and rerun.")


if __name__ == "__main__":
    main()
