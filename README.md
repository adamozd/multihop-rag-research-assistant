# Multi-Hop RAG Research Synthesis Assistant

This is a local research prototype developed by me that searches academic papers, writes cited claims, and checks them against the evidence. The default flow combines drafting with evidence assessment, then reviews and corrects the answer in one more call. The original multi-step research flow remains available for evaluation.

The starter corpus covers **RAG and multi-hop question answering**. The pipeline uses plain Python; embeddings and Chroma stay local. All LLM calls, including JSON repair and revision, pass through `llm_call(prompt: str) -> str` in `llm/watsonx_client.py`.

## Quick start

Use Python **3.12** (the tested version). Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Set `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, and `WATSONX_MODEL_ID` in your local `.env`. Never commit credentials. The example uses Toronto and `meta-llama/llama-3-3-70b-instruct`, confirmed in this project's regional catalog. Choose an instruction model that supports watsonx's chat API; availability depends on the region and account. The originally proposed Granite 3.3 model is unavailable in this setup. Validate the actual configuration before inference:

```bash
python -m llm.watsonx_client
python -m ingestion.fetch_papers
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal, activate the same environment and run:

```bash
streamlit run frontend/app.py --server.address 127.0.0.1
```

Open Streamlit at `http://127.0.0.1:8501`; API documentation is at `http://127.0.0.1:8000/docs`. These are local development services, without authentication or production deployment configuration.

For VS Code, open this repository's root folder and install the recommended Python and Python Debugger extensions. In **Run and Debug**, select **Research Assistant: Full app** and press **F5**. The included launch configurations explicitly use `.venv` for Python, load `.env`, and run from the project root. You can also select **Research Assistant: Validate watsonx** to check provider access. Restart both processes after editing `.env`; the model client is cached. Running `api/main.py` or `frontend/app.py` directly with the editor's generic “Run Python File” button does not start the required Uvicorn/Streamlit servers.

Ingestion downloads the 15 curated papers in `ingestion/corpus.json` using the arXiv API and downloads the BGE model on first use. PDFs, model files, metadata and the persistent index are stored under ignored `data/`. Re-running ingestion reuses versioned PDFs and replaces stale chunks for each paper. Do not run indexing concurrently with evaluation. To use a different corpus:

```bash
python -m ingestion.fetch_papers --query 'all:"retrieval augmented generation"' --limit 20
```

Use a **new `CHROMA_COLLECTION`** when switching corpora; existing papers are otherwise retained. Adapt the evaluation questions and expected-paper labels to match. `--limit` applies to topic search; curated mode always fetches the 15 specified IDs.

After the initial model download, set `HF_HUB_OFFLINE=1` to prevent Hugging Face metadata requests. To rebuild the index after changing chunking without refetching papers: `HF_HUB_OFFLINE=1 python -m ingestion.fetch_papers --reindex-local`.

## Your papers and evidence explorer

In the Streamlit sidebar, open **Manage papers**, enter a collection name, and select **Create collection**. The new collection is selected automatically. Upload one or more PDFs and choose **Index PDFs**. Once indexing finishes, ask a question: retrieval searches only the selected collection. Switching collections clears the previous answer to avoid mixing sources.

Uploads support text-based PDFs up to **20 MB and 200 pages each**. Password-protected and image-only PDFs are rejected; OCR is not included. Re-uploading identical bytes to the same collection does not duplicate the paper. Uploads are stored locally and embedded with BGE; indexing does not call watsonx. Asking questions still sends retrieved text to your configured watsonx model and requires available quota. English papers are the intended fit for the current embedding model.

The existing **RAG & multi-hop QA** corpus remains the default and is read-only through the upload interface. Custom collections have separate Chroma indexes. Their catalog and PDFs live under ignored `data/library/` (override with `LIBRARY_PATH`); vectors remain under `CHROMA_PATH`. This is local collection separation, not account-based access control. Keep the unauthenticated API bound to localhost.

In **Answer**, expand **Explore evidence** for a claim to see each exact cited excerpt, title, section and PDF page numbers. Select **Show source page** to view the original PDF page and its extracted text, including surrounding context. You can change the page number to inspect adjacent pages. This extra context is for your review; it is not automatically added to the saved answer's evidence or sent to the model. Missing citations or unavailable source PDFs are shown explicitly. The same page viewer is available under **Sources**.

Collection/source endpoints:

- `GET /collections` and `POST /collections` with `{"name":"My research"}`.
- `GET /collections/{id}/papers` lists indexed papers.
- `POST /collections/{id}/papers?filename=paper.pdf` accepts raw PDF bytes (`Content-Type: application/pdf`).
- `GET /collections/{id}/papers/{paper_id}/pdf` serves the stored original.
- `GET /collections/{id}/papers/{paper_id}/pages/{page}` returns page text; append `/image` for a rendered PNG.
- `POST /ask` accepts an optional `collection_id`; it defaults to `"default"` for existing clients and evaluation scripts. Results record the collection ID.

Restart both the API and Streamlit after updating. No additional dependencies are needed. Live stage-by-stage progress is a later change; current indexing and synthesis use ordinary spinners.

## Request and output

```bash
curl http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do IRCoT and Self-RAG decide when to retrieve evidence?","mode":"efficient","critique":true,"max_hops":2,"top_k":4}'
```

`POST /ask` returns structured draft and final claims, rendered answers, source excerpts with paper/section/PDF-page provenance, retrieval decisions, critique verdicts, and warnings. The Streamlit interface exposes the draft, final answer, citations and collapsible traces. `critique=false` returns the unaudited draft for interactive comparison.

### Efficient mode (default)

1. Search the selected collection locally using the original question.
2. In one model call, draft cited claims and identify any essential missing evidence.
3. If needed, make one targeted local search. Only new usable evidence triggers another drafting call.
4. In one model call, audit the draft and correct or remove flagged claims. Supported claims are kept unchanged; new claim IDs and unresolved citations are excluded.

Normally this uses **2 model calls**, or **3 with a useful follow-up**, before JSON repairs. An abstention needs no review call. The review log evaluates the draft; the corrected final answer is **not independently re-audited**. This cost/quality trade-off has been tested with mocked responses, not yet with live watsonx. No measured token savings or equal-quality claim is made.

Efficient mode sends at most **4 complete chunks / 8,000 evidence-text characters** at a time. It skips oversized chunks instead of shortening them into potentially misleading snippets. Selection still depends on vector ranking; narrower context may miss useful evidence. `top_k` is capped at 4 and `max_hops` at 2 in this mode.

Per request, the shared client permits at most **6 inference attempts including JSON repairs**, at most **32,000 characters per user prompt including schema/repair text**, and at most **1,800 output tokens per call** (or your lower `WATSONX_MAX_NEW_TOKENS`). These are call/context/output limits, not a guaranteed monetary or total-token budget. Too-short outputs can still fail JSON validation.

Identical efficient requests reuse successful results for one hour, with up to 32 entries in process memory. Cache keys include the question, collection, request settings, indexed text/metadata fingerprint, model, region, project and output setting. Restarting the API clears the cache. Use `use_cache=false` to bypass it; baseline requests always bypass it. Do not index concurrently with synthesis or evaluation.

The UI and result JSON show actual provider-reported input/output tokens and inference attempts, including repairs. Cache hits show zero new usage. If a response omits usage or a call fails without counts, totals are marked unavailable. The local, ignored `data/usage.jsonl` ledger also records failed requests; it contains counts and statuses, never questions, excerpts, credentials or raw responses. Override its path with `LLM_USAGE_LOG`. These totals are application records, not a replacement for IBM's account-wide usage report.

SDK retries are disabled. On `token_quota_reached`, the client blocks further inference for the life of that process. Restore quota before restarting the API. A restart only clears the local block; it does not restore the provider allowance. Benchmark and acceptance runners stop immediately on quota exhaustion and mark remaining work as unattempted.

### Research baseline

Select **Research baseline** in the sidebar or send `mode=baseline`. This preserves the original reasoning sequence and up-to-24-chunk context for comparisons. It uses the configured output limit and has a 16-attempt safety ceiling, including repairs. Both evaluation runners explicitly select this mode with caching disabled.

1. Decompose into 2–4 atomic questions.
2. Retrieve top-k chunks for each subquestion. This initial batch counts as **hop 1**.
3. Ask the controller whether evidence suffices. Additional hops retrieve one targeted query each. Maximum 1–3 total hops; default 2. Repeated queries stop the loop.
4. Deduplicate evidence by chunk ID and cap context at 24 chunks, reserving capacity for later hops. Round-robin selection preserves subquestion coverage under the cap.
5. Synthesize atomic claims, each with chunk citation IDs; contradictions must also be cited claims. An answer can be partial or abstain when evidence is missing.
6. Audit every claim for entailment, including scope, numerical results and comparisons. Reject incomplete/duplicate critic coverage; deterministically flag invalid citations even if the model calls them supported.
7. If any claim is flagged, revise once and audit again. **Remaining problems stay visible in the critique log and warnings.** This is not a guarantee that the final answer is correct.

All structured model outputs receive at most **one repair attempt**, including schema errors. On a JSON parse error, the repair prompt includes the required text: “your last response was not valid JSON — return only the JSON object”. A second failure aborts the request with an explicit error. Provider failures are surfaced, never replaced by simulated research answers.

The section-aware chunker detects common and numbered headings, packs paragraphs within sections, and only splits oversized text to a 480-token budget using BGE's tokenizer. It preserves section and page provenance and avoids combining sections. Heading detection is heuristic; complex two-column layouts, tables, scanned PDFs and unusual headings require manual inspection. OCR is not implemented. Retrieved material is explicitly treated as untrusted data in reasoning prompts.

The shared LLM client uses watsonx's chat endpoint so instruction models receive their conversation formatting. All current reasoning calls expect JSON, so the client requests `response_format={"type": "json_object"}` and supplies a JSON-only system instruction. JSON mode does not guarantee schema correctness: strict validation and the single repair attempt remain mandatory. `WATSONX_MAX_NEW_TOKENS` maps to the chat parameter `max_tokens`; temperature is zero. Empty responses report a sanitized finish reason. Structured-output failures name the expected schema and report syntax positions or field/type errors without logging raw model responses. Catalog validation does not test generation: after changing models or client code, restart the backend and submit a real question.

## Evaluation

`eval/eval_questions.json` contains 18 starter questions: 5 single-hop controls, 9 multi-hop comparisons and 4 adversarial/no-answer cases. The first live run that I tested uses the frozen snapshot in `eval/benchmarks/v1/questions.json`, with its hash recorded in `manifest.json`. Expected-paper and abstention labels were frozen before inference but are **not independently human-validated**. Reference answers are for reviewers, never supplied to the answering pipeline. This is a development benchmark; the comparison question was used during development.

```bash
HF_HUB_OFFLINE=1 python -m eval.run_eval --questions eval/benchmarks/v1/questions.json
HF_HUB_OFFLINE=1 python -m eval.run_acceptance
```

Be careful that these commands consume watsonx quota. **The current runner has no resume support**: invoking it again starts a new run. Preserve existing results and add verified resume support before retrying the failed questions. See [human-review instructions](eval/HUMAN_REVIEW.md).

Each question runs once with critique enabled. Its exact original draft is the “without critique” arm and its final answer is the “with critique” arm. Both arms therefore share the same retrieval, draft and evidence. This isolates the revision loop from retrieval and sampling changes. It does not measure an independent retrieve-once baseline.

Timestamped output directories contain full per-question results, `run.json`, and shuffled `human-review.json`. Errors are recorded explicitly, checkpointed after every question, and cause a nonzero exit status. Do not report partial runs as complete experiments. The run records model, region, request settings, question hash and corpus-manifest hash; archive the environment lock and source revision with reported experiments.

Metrics:

- **Claim-support rate:** supported claims / all emitted claims, pooled across successful questions, separately for each arm. Zero claims yields `null`, never a perfect score.
- **Hop recall:** unique expected papers cited / unique expected papers, averaged over multi-hop questions. Retrieved-but-uncited papers do not count. This measures paper coverage, not correctness of the reasoning chain.
- **Answerable response rate** and **no-answer abstention rate:** reported alongside support to expose gains obtained simply by deleting claims or abstaining.

The automatic report is labeled **internal critic proxy, not independent ground truth**. Using the same LLM as generator and judge can reinforce its mistakes. For a defensible headline, give `human-review.json` to a reviewer without the arm mapping in `run.json`; fill each `label` with `supported`, `unsupported`, or `contradicted`, then score:

```bash
python -m eval.run_eval \
  --score-run eval/results/RUN_ID/run.json \
  --annotations eval/results/RUN_ID/human-review.json
```

Support means the **entire claim** is entailed by its cited excerpts. Missing/invalid citations, unsupported specificity and overstated causality fail support. Review claim atomicity too: compound claims should be split before the benchmark is frozen. Limitations are restricted by prompt to missing evidence/scope and excluded from claim scoring; a reviewer must check that factual assertions have not leaked there. For stronger reporting, use two independent reviewers and resolve disagreements. The scorer requires complete labels and matching run IDs, and writes a separate `human-report.json`. No benchmark improvement is claimed before actual evaluation.

## Development and structure

```text
ingestion/    arXiv fetch, PDF parsing, section-aware chunks, curated corpus
retrieval/    local BGE embeddings and persistent Chroma
llm/          isolated watsonx integration and shared JSON retry
reasoning/    schemas, decomposition, hop controller, synthesis, critic, pipeline
eval/         questions, paired evaluation, metrics, human annotation scoring
api/          FastAPI POST /ask
frontend/     Streamlit interface
tests/        offline behavior and local integration tests
```

```bash
python -m pytest -q
pip check
```

Tests exercise JSON/schema repair, the shared provider boundary, bounded hops and revision, citation integrity, critic completeness, abstention, metrics, real PDF extraction, persistent Chroma with deterministic test embeddings, and API validation. They do not require credentials or model downloads, and do not establish model answer quality. `requirements.lock.txt` records the installed Python 3.12/macOS environment; `requirements.txt` gives compatible dependency ranges for other platforms.

Live synthesis and model validation require your configured watsonx project. The SDK validates the model in that region; no silent fallback is used. Request state is isolated; calls on the shared SDK session are serialized. This prototype is optimized for a local research workflow rather than multi-user throughput.

Local verification: **77 offline tests pass**, covering request limits, token accounting, caching, quota fail-fast, both reasoning flows, PDF upload validation, duplicate handling, collection isolation, failed-index cleanup, empty-collection quota protection, and evidence-page UI behavior. PDF regression cases cover shuffled two-column content, single-column text, wrapped headings, numeric table rows, repeated margins and appendices following references. Streamlit tests also render the user's saved research result, numbered sources, review states and connection errors. The collection controls and claim-to-PDF explorer were checked in the browser using isolated local storage and a saved answer. A real 22-page upload produced 50 searchable passages and a verified page preview without watsonx calls.

The current extraction uses paragraph blocks and inferred column boundaries rather than line-by-line coordinate sorting. Heading detection uses font/layout information; repeated margin text is removed. It remains heuristic: complex tables, equations, unusual layouts and scanned PDFs require manual review. The local 15-paper corpus has been rebuilt into 695 chunks (maximum 480 BGE tokens); `data/qa/verification.json` records chunk counts, the 480-token check and focused retrieval smoke checks. Both focused smoke queries retrieved their expected paper in the top four; this measures paper presence, not complete evidence coverage. Older exports keep their old evidence IDs; rerun questions to inspect the new evidence.

Decomposition now restricts itself to the requested facet. Every insufficient hop decision must provide a concrete `missing_fact`; sufficient decisions require null query and missing fact. Repeated queries and hops without new evidence stop with an explicit reason in the trace. Synthesis and critique explicitly check qualifiers and alternatives, including the optional Self-RAG threshold distinction missed in the first live result. All calls still use the shared watsonx client and one JSON/schema repair attempt.

To inspect the refreshed interface, restart the API and Streamlit using the startup commands above (or the VS Code Full app configuration). Search settings are in the sidebar. Results have Answer, Sources, Review and Retrieval tabs; numbered paper citations map to source excerpts, while the downloaded JSON preserves the original evidence IDs.

## Live evaluation status — 2026-10-05

Live watsonx testing **has been performed**, using Llama 3.3 70B Instruct in Toronto. The frozen 18-question benchmark was attempted; run `20261005T230218Z-ecad21ea` completed **11/18 questions** before token-quota exhaustion blocked the remainder. Provider logs also recorded one request-rate-limit response. A completed execution is not necessarily a correct or complete answer.

| Group | Completed | Planned |
| --- | ---: | ---: |
| Single-hop | 5 | 5 |
| Multi-hop | 6 | 9 |
| Adversarial/no-answer | 0 | 4 |

The separate acceptance run `acceptance-20261005T230751Z` completed **0/3 probes** because quota was exhausted. These checks are blocked, not passed. An earlier user-run comparison did exercise live revision, but review found unsupported assertions surviving revision and inconsistent critic judgments.

Across the 11 completed benchmark questions, the **internal critic proxy** changed from 33/36 supported claims (91.67%) to 32/33 (96.97%). Mean expected-paper recall across the six completed multi-hop cases fell from 83.33% to 75.00%; the answerable response rate was 10/11 in both arms. No-answer abstention was not measured. Two questions triggered revision. Fewer claims and lower paper recall mean the higher proxy rate must not be presented as proven answer-quality improvement.

**Independent human-reviewed claim support has not been established.** The saved labels were completed and submitted by the author after AI assistance; they must not be presented as an independent human evaluation. The full benchmark and semantic acceptance remain incomplete. [Sanitized results and limitations](eval/reports/README.md) are committed separately from ignored raw `eval/results/` files.

## Primary references that I used for implementation

- [IBM ModelInference SDK](https://ibm.github.io/watsonx-ai-python-sdk/v1.4.11/fm_model_inference.html) and [supported model catalog](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&locale=en)
- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5): local normalized embeddings and retrieval query instruction
- [Chroma Python client](https://docs.trychroma.com/reference/python/client): persistent local storage
- Corpus IDs link to primary papers as `https://arxiv.org/abs/ID`; downloaded metadata records versioned source URLs.
