# Next steps

Checkpoint: 2026-10-05. Steps 1–3 are implemented and live watsonx evaluation has begun. The first frozen benchmark attempt completed 11/18 questions; quota exhaustion blocked completion and all three separate acceptance probes. Human review remains pending.

## 1. Evidence extraction — implemented, locally verified

PDF extraction now orders paragraph blocks by column, uses spanning blocks as reading boundaries, removes repeated margin text, and verifies section labels using layout/font information. Numeric table rows stay in the evidence rather than becoming section titles. Page and paper provenance and the 480-token cap are retained. The 15-paper local corpus has been rebuilt into 695 chunks, all at or below 480 BGE tokens.

Representative IRCoT and Self-RAG passages were compared with rendered PDF pages. Regression tests cover shuffled two-column objects, full-width titles, single-column text, wrapped headings, table rows and repeated headers. Complex tables, equations, unusual layouts and scanned PDFs remain limitations; this is not a general document-layout engine.

## 2. Focused retrieval — implemented, live acceptance pending

Decomposition prompts restrict questions to the requested facet. The controller must name a concrete missing fact before another search, and ignore decomposition scope creep. Repeated queries and hops with no new evidence stop retrieval; the trace records the stop reason and any unresolved fact.

An earlier live comparison kept decomposition focused but missed the key Self-RAG inference passage and used a near-paraphrase for its follow-up search. Semantic acceptance remains open: the IRCoT/Self-RAG timing comparison does not invent an efficiency requirement and stops when both requested mechanisms are supported. Prompt instructions alone do not guarantee this behavior.

## 3. Synthesis and claim auditing — implemented, live acceptance pending

Synthesis and critic instructions now explicitly distinguish optional from mandatory behavior, alternative policies from combined steps, training from inference, and dataset-specific results from general claims. The critic must explain decisive evidence wording. Revision remains capped at one, with unresolved flags visible.

Offline regression verifies the correction flow; live revision has also executed. An earlier user-run comparison retained unsupported ideas after revision and produced inconsistent critic judgments. Two of the 11 completed benchmark questions triggered revision, but this is not evidence of improved human-reviewed support. The separate comparison, no-answer and seeded-error acceptance probes all failed to complete after quota exhaustion. Repeat them when quota is available and obtain human semantic judgments.

## UI — initial cleanup implemented

A restrained Streamlit theme, sidebar search settings, numbered paper citations, expandable excerpts, and separate Answer / Sources / Review / Retrieval tabs replace the dense output. Drafts, unresolved warnings, stop reasons and JSON export remain available. Offline Streamlit tests cover success, abstention, errors and the user's saved result; browser visual review is still pending.

## 4. Complete the frozen baseline — quota blocked, human review pending

The v1 questions, expected papers and abstention labels are frozen under `eval/benchmarks/v1/`; independent human validation is still pending. Run `20261005T230218Z-ecad21ea` completed s01–s05 and m01–m06. Failures: m07–m09 and a01–a04. Preserve the raw results and blinded review file locally.

Partial internal metrics: 33/36 supported draft claims versus 32/33 final claims (91.67% to 96.97%); expected-paper recall declined from 83.33% to 75.00%; answerable response rate was 10/11 in both arms. No adversarial case completed, so no-answer abstention is unmeasured. All 69 human-review items are unlabeled. These are internal model judgments on a partial development benchmark, not human-reviewed accuracy or proof of improvement. See [public summary](eval/reports/README.md).

Next actions, in order:

1. Add quota-aware stopping and resume support. The current runner starts fresh; it must preserve successful results, verify question/source/corpus/configuration compatibility and record resumed attempts instead of selecting favorable reruns.
2. Confirm quota availability before making new inference calls. Complete the seven failed benchmark questions and all three separate acceptance probes; retain failure history and disclose changes between sessions.
3. Obtain real human labels for the shuffled claim-review file. Review can begin on the 11 completed questions now. Keep the draft/final mapping hidden from reviewers and record reviewer provenance.
4. Score complete annotations and report support, claim counts, paper recall, answerable response rate, no-answer abstention and failures together. Separate any partial report from a completed benchmark report.
5. Add token-usage/cost instrumentation before further scaling. Per-question durations are recorded; token usage is not yet captured.

Publishing: commit evaluation code, frozen questions, review instructions and sanitized summaries under `eval/reports/`. Keep raw `eval/results/`, provider logs, reviewer working files, `.env`, PDFs and the local index out of Git. A sanitized raw-artifact release can be considered separately after review.

Acceptance: a complete, reproducible report with human-reviewed labels and limitations, whether or not critique improves the headline score. This milestone has not yet been reached.

## 5. Expand scope after quality is established

Add PDF uploads and isolated document collections, then topic-based paper discovery with user review of selected papers. Plan a public beta after reliability and usage costs are understood; include access controls, usage limits, private collection isolation and deployment monitoring.
