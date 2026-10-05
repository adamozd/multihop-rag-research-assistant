# Next steps

Checkpoint: 2026-10-05. Development is paused after the first successful live answer.

## Current baseline

- Local Streamlit interface and FastAPI backend; 15 papers indexed into 780 chunks.
- watsonx in Toronto using Llama 3.3 70B Instruct, with chat JSON mode and one schema/parse repair attempt.
- A user-run result for the IRCoT/Self-RAG comparison completed decomposition, two retrieval hops, synthesis and critique.
- The result contained two claims, valid citations to both expected papers, and no revision because the critic accepted both claims.
- Manual review found an imprecise claim the critic missed: Self-RAG's optional threshold policy was described as though it necessarily accompanied the retrieval-token decision.
- Some PDF excerpts interleave columns or misclassify table rows as section headings. Decomposition introduced an efficiency trade-off beyond the question, contributing to an unnecessary evidence-insufficiency warning.
- No benchmark improvement has been demonstrated. A successful run is evidence of integration, not proof of answer quality or revision effectiveness.

## 1. Fix evidence extraction

Improve reading order for multi-column PDFs and heading detection while preserving paper, section and page provenance. Re-index the existing corpus after changes.

Acceptance: manually compare representative IRCoT and Self-RAG passages with their PDF pages; ensure columns are not interleaved and table rows are not headings. Add regression fixtures for those layouts and retain the token cap.

## 2. Keep retrieval focused on the user's question

Restrict decomposition to necessary subquestions. Have the controller identify a concrete missing fact and target it, rather than broadening the research scope or repeating the original question.

Acceptance: the retrieval-timing comparison does not invent an accuracy/efficiency requirement. It stops when the requested mechanisms are supported and accurately reports any remaining gap.

## 3. Improve synthesis and claim auditing

Preserve qualifiers such as optional versus mandatory, alternatives versus combined steps, and dataset-specific versus general results. Strengthen the critic with focused regression cases, including the missed Self-RAG distinction. Keep the one-revision cap and expose unresolved problems.

Acceptance: human review confirms that deliberately overstated claims are flagged and corrected or removed. Verify the revision path in a live run as well as with mocked tests. Add a no-answer case to check abstention.

## 4. Establish a defensible evaluation baseline

Review and freeze the 18 starter questions and expected-paper labels. Run a small pilot, inspect token usage, then run the paired draft/final evaluation. Record model, corpus version and configuration; obtain independent human support labels.

Report claim-support rate before/after critique, cited-paper recall, claim counts, answerable response rate, no-answer abstention rate and failures. Track latency and token usage before planning public access. Do not present the internal critic's ratings as independent ground truth.

Acceptance: a complete, reproducible report with human-reviewed labels and limitations, whether or not critique improves the headline score.

## 5. Expand scope after quality is established

Add PDF uploads and isolated document collections, then topic-based paper discovery with user review of the selected papers. Plan a public beta only after reliability and usage costs are understood; include accounts or access controls, usage limits, private collection isolation and deployment monitoring.

The next development session should start with step 1. These are planned changes, not implemented features.
