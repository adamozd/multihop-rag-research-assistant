# Next steps

Checkpoint: 2026-10-05. Steps 1–3 are implemented locally; model-quality acceptance still needs a live run and human review.

## 1. Evidence extraction — implemented, locally verified

PDF extraction now orders paragraph blocks by column, uses spanning blocks as reading boundaries, removes repeated margin text, and verifies section labels using layout/font information. Numeric table rows stay in the evidence rather than becoming section titles. Page and paper provenance and the 480-token cap are retained. The 15-paper local corpus has been rebuilt into 695 chunks, all at or below 480 BGE tokens.

Representative IRCoT and Self-RAG passages were compared with rendered PDF pages. Regression tests cover shuffled two-column objects, full-width titles, single-column text, wrapped headings, table rows and repeated headers. Complex tables, equations, unusual layouts and scanned PDFs remain limitations; this is not a general document-layout engine.

## 2. Focused retrieval — implemented, live acceptance pending

Decomposition prompts restrict questions to the requested facet. The controller must name a concrete missing fact before another search, and ignore decomposition scope creep. Repeated queries and hops with no new evidence stop retrieval; the trace records the stop reason and any unresolved fact.

Acceptance still to verify with the configured model: the IRCoT/Self-RAG timing comparison does not invent an efficiency requirement and stops when both requested mechanisms are supported. Prompt instructions alone do not guarantee this behavior.

## 3. Synthesis and claim auditing — implemented, live acceptance pending

Synthesis and critic instructions now explicitly distinguish optional from mandatory behavior, alternative policies from combined steps, training from inference, and dataset-specific results from general claims. The critic must explain decisive evidence wording. Revision remains capped at one, with unresolved flags visible.

An offline regression verifies the pipeline's correction flow for an overstated retrieval policy using scripted model outputs. This does **not** demonstrate that the live model detects the error. Next: rerun the comparison, manually check the optional Self-RAG threshold, exercise a real revision, and try a no-answer question.

## UI — initial cleanup implemented

A restrained Streamlit theme, sidebar search settings, numbered paper citations, expandable excerpts, and separate Answer / Sources / Review / Retrieval tabs replace the dense output. Drafts, unresolved warnings, stop reasons and JSON export remain available. Offline Streamlit tests cover success, abstention, errors and the user's saved result; browser visual review is still pending.

## 4. Establish a defensible evaluation baseline

Review and freeze the 18 starter questions and expected-paper labels. Run a small pilot, inspect token usage, then run the paired draft/final evaluation. Record model, corpus version and configuration; obtain independent human support labels.

Report claim-support rate before/after critique, cited-paper recall, claim counts, answerable response rate, no-answer abstention rate and failures. Track latency and token usage before planning public access. Do not present the internal critic's ratings as independent ground truth.

Acceptance: a complete, reproducible report with human-reviewed labels and limitations, whether or not critique improves the headline score.

## 5. Expand scope after quality is established

Add PDF uploads and isolated document collections, then topic-based paper discovery with user review of selected papers. Plan a public beta after reliability and usage costs are understood; include access controls, usage limits, private collection isolation and deployment monitoring.
